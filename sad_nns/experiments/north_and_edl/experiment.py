#!/usr/bin/env python3
from copy import copy

from neurops import *
import numpy as np
import torch
from torch.functional import F
from sad_nns.experiments.north_and_edl import config
from sad_nns.experiments.data import Dataset
from sad_nns.uncertainty import KLDivergenceLoss
from skimage.util import random_noise
from torchvision import transforms
import wandb

device = config.device
kl_divergence = KLDivergenceLoss()

DO_GROW_PRUNE = False

BASIC_TABLE_COLUMNS = ["_id", "image", "prediction", "target", "uncertainty"]
NUM_BATCHES_TO_LOG = 10
NUM_IMAGES_PER_BATCH = 32

def train(model, train_loader, optimizer, criterion, epochs, num_classes,
          val_loader=None, verbose=True, training_cycle=0):
    model.train()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Convert target to one-hot encoding
            target_one_hot = F.one_hot(target, num_classes=num_classes)

            optimizer.zero_grad()
            output = model(data)

            # Calculate criterion loss
            evidence = F.relu(output)
            criterion_loss = criterion(evidence, target_one_hot)

            # Calculate KL Divergence Loss
            kl_divergence_loss = kl_divergence(evidence, target_one_hot)
            annealing_step = 10
            annealing_coef = torch.min(
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(epoch / annealing_step, dtype=torch.float32)
            )

            # Calculate total loss
            loss = criterion_loss + annealing_coef * kl_divergence_loss
            loss.backward()
            optimizer.step()

             # Calculate uncertainty
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            mean_uncertainty = torch.mean(uncertainty)

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(data)

            # Log statistics
            if batch_idx % 50 == 0:
                if verbose:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), accuracy))

                wandb.log({
                    "training_cycle": training_cycle,
                    "epoch": epoch,
                    "train/batch_idx": batch_idx,
                    "train/loss": loss.item(),
                    "train/criterion_loss": criterion_loss.item(),
                    "train/kl_divergence_loss": kl_divergence_loss.item(),
                    "train/uncertainty": mean_uncertainty.item(),
                    "train/accuracy": accuracy,
                })

        # Validate
        if val_loader is not None:
            print("Validation: ", end = "")
            statistics = test(model, val_loader, criterion, num_classes)

            statistics = {f"val/{key}": value for key, value in statistics.items()}
            wandb.log({"training_cycle": training_cycle, "epoch": epoch, **statistics})


def test(model, test_loader, criterion, num_classes, log_table=None):
    model.eval()
    test_loss = 0
    correct = 0
    uncertainties = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Convert target to one-hot encoding
            data, target = data.to(device), target.to(device)
            one_hot_target = F.one_hot(target, num_classes=num_classes)

            # Calculate loss
            output = model(data)
            evidence = F.relu(output)
            test_loss += criterion(evidence, one_hot_target).item()

            # Get predictions and calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Calculate uncertainty
            alpha = evidence + 1
            u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            uncertainties.append(u.mean())

            # Log predictions
            if log_table is not None and batch_idx < NUM_BATCHES_TO_LOG:
                log_data = data.cpu().numpy()
                log_pred = pred.cpu().numpy()
                log_target = target.cpu().numpy()
                log_u = u.cpu().numpy()
                log_evidence = evidence.cpu().numpy()
                for i in range(NUM_IMAGES_PER_BATCH):
                    id = i + batch_idx * NUM_IMAGES_PER_BATCH
                    log_table.add_data(
                        id,
                        wandb.Image(log_data[i]),
                        log_pred[i],
                        log_target[i],
                        log_u[i],
                        *log_evidence[i],
                    )

    # Calculate statistics
    test_loss /= batch_idx + 1
    accuracy = correct / len(test_loader.dataset)
    avg_u = torch.mean(torch.stack(uncertainties))

    # Log statistics
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Average Uncertainty: {:.4f}'.format(
        test_loss, correct, len(test_loader.dataset), accuracy, avg_u))

    rtn = {
        "loss": test_loss,
        "accuracy": accuracy,
        "uncertainty": avg_u
    }

    if log_table is not None:
        rtn.update({"predictions": log_table})

    return rtn


def grow(model, optimizer):
    grow_metric = config.grow_metric
    metric_params = config.grow_metric_params
    tensor_func = metric_params.pop('tensor')

    stats = {
        "num_grown": 0,
    }

    for i in range(len(model)-1):
        tensor = tensor_func(model, i)
        score = grow_metric(tensor, **metric_params)

        max_rank = model[i].width()
        to_add = max(score - int(0.95 * max_rank), 0)

        stats["num_grown"] += to_add
        print("Layer {} score: {}/{}, neurons to add: {}".format(i, score, max_rank, to_add))
        model.grow(i, to_add, fanin_weights="iterative_orthogonalization", optimizer=optimizer)

    # Add tensor_func back to metric_params for next loop
    metric_params['tensor'] = tensor_func

    return stats


def prune(model, optimizer):
    prune_metric = config.prune_metric
    metric_params = config.prune_metric_params
    tensor_func = metric_params.pop('tensor')

    stats = {
        "num_pruned": 0,
    }

    # Prune the model
    for i in range(len(model)-1):
        tensor = tensor_func(model, i)
        scores = prune_metric(tensor, **metric_params)
        to_prune = np.argsort(scores.detach().cpu().numpy())[:int(0.25*len(scores))]

        stats["num_pruned"] += len(to_prune)
        print("Layer {} scores: mean {:.3g}, std {:.3g}, min {:.3g}, smallest 25%: {}".format(
            i, scores.mean(), scores.std(), scores.min(), to_prune))

        model.prune(i, to_prune, optimizer=optimizer, clear_activations=True)

    # Add tensor_func back to metric_params for next loop
    metric_params['tensor'] = tensor_func

    return stats


if __name__ == '__main__':
    # Initialize config
    train_loader = config.dataset.train_loader
    val_loader = config.dataset.val_loader
    test_loader = config.dataset.test_loader
    num_classes = config.dataset.num_classes
    epochs = config.epochs
    criterion = config.criterion

    # Initialize model
    model = ModSequential(
        ModConv2d(in_channels=1, out_channels=8, kernel_size=7, masked=True, padding=1, learnable_mask=True),
        ModConv2d(in_channels=8, out_channels=16, kernel_size=7, masked=True, padding=1, prebatchnorm=True, learnable_mask=True),
        ModConv2d(in_channels=16, out_channels=16, kernel_size=5, masked=True, prebatchnorm=True, learnable_mask=True),
        ModLinear(64, 32, masked=True, prebatchnorm=True, learnable_mask=True),
        ModLinear(32, num_classes, masked=True, prebatchnorm=True, nonlinearity=""),
        track_activations=True,
        track_auxiliary_gradients=True,
        input_shape = (1, config.dataset.image_size, config.dataset.image_size)
    ).to(device)
    torch.compile(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # Initialize wandb
    wandb.init(
        project="north-and-edl",
        config=config.config,
        # mode="disabled"
    )

    # Train
    print("Performing initial training of model...")
    train(model, train_loader, optimizer, criterion, epochs, num_classes,
          val_loader=val_loader, training_cycle=0)

    # Grow and prune
    for train_cycle in range(config.grow_prune_cycles):
        if not DO_GROW_PRUNE:
            break

        print(f"Grow and prune cycle {train_cycle+1} of {config.grow_prune_cycles}...")

        # Grow or prune
        grow_or_prune = next(config.grow_prune_strategy)
        if grow_or_prune == 'grow':
            grow_stats = grow(model, optimizer)
            wandb.log({'cycle': train_cycle+1, 'action': 'grow', **grow_stats})
        else:
            prune_stats = prune(model, optimizer)
            wandb.log({'cycle': train_cycle+1, 'action': 'prune', **prune_stats})

        # Train and validate
        print(f"Validation after {grow_or_prune}: ", end = "")
        val_stats = test(model, val_loader, criterion, num_classes)
        val_stats = {f"val/{key}": value for key, value in val_stats.items()}
        wandb.log({'cycle': train_cycle+1, "epoch": -1, **val_stats})
        train(model, train_loader, optimizer, criterion, config.epochs_per_cycle, num_classes,
              val_loader=val_loader, training_cycle=train_cycle+1)

    # Test
    columns = copy(BASIC_TABLE_COLUMNS)
    for class_name in config.dataset.classes:
      columns.append("evidence_" + str(class_name))
    test_table = wandb.Table(columns=columns)

    print("Testing model...")
    test_stats = test(model, test_loader, criterion, num_classes, log_table=test_table)

    print("Test statistics: ", test_stats)
    test_stats = {f"test/{key}": value for key, value in test_stats.items()}
    wandb.log(test_stats)

    # Noisy MNIST
    columns = copy(BASIC_TABLE_COLUMNS)
    for class_name in config.dataset.classes:
      columns.append("score_" + str(class_name))
    noise_table = wandb.Table(columns=columns)

    variances = [0.005, 0.01, 0.05, 0.1]

    print("Testing model on noisy data...")
    for var in variances:
        print(f"Testing model on noise variance {var}...")

        noise_transform = transforms.Lambda(lambda x: torch.tensor(random_noise(x, mode='s&p', amount=var, clip=False), dtype=torch.float32)) # choose speckle, change amount ot var
        noise_dataset = Dataset(
            config.dataset.dataset_name,
            config.dataset.image_size,
            config.dataset.batch_size,
            split=0.9,
            extra_transforms=[noise_transform]
        )
        noise_test_loader = noise_dataset.test_loader

        noise_stats = test(model, noise_test_loader, criterion, num_classes, log_table=noise_table)
        noise_stats.update({"noise_variance": var})
        noise_stats = {f"noise/{key}": value for key, value in noise_stats.items()}
        wandb.log(noise_stats)

