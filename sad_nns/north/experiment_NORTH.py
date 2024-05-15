# Our setup
import torch
from torch.functional import F

# change to "cuda:x" to switch gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu" if not torch.cuda.is_available() else "cuda:1"

import copy
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from sad_nns.uncertainty import *
from neurops import *

import random

# Generate seed
seed = random.randint(0,1000)
torch.manual_seed(seed)
print("Seed of this run: ", seed)

# parameters
batch_size = 512
epsilon = 0.01
growth_epoch = 12
end_epoch = 3
total_epoch = 15
learning_rate = 0.01
train_accs= []
test_accs = []
epochs = np.arange(total_epoch)

model = ModSequential(
        # ModConv2d(in_channels=1, out_channels=8, kernel_size=7, padding=1),
        # ModConv2d(in_channels=8, out_channels=16, kernel_size=7, padding=1),
        # ModConv2d(in_channels=16, out_channels=16, kernel_size=5),
        # ModLinear(64, 32),
        # ModLinear(32, 10, nonlinearity=""),
        ModLinear(784, 64),
        ModLinear(64, 64),
        ModLinear(64, 64),
        ModLinear(64, 10,nonlinearity=""),
        track_activations=True,
        track_auxiliary_gradients=False,
        # input_shape = (1, 14, 14)
        input_shape = (784) # TODO: change accordingly
    ).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# load dataset
dataset = datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([ 
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Resize((14,14)) #TODO: change accordingly
                            transforms.Resize((28,28)),
                            transforms.Lambda(lambda x: torch.flatten(x))
                        ]))
train_set, val_set = torch.utils.data.random_split(dataset, lengths=[int(0.9*len(dataset)), int(0.1*len(dataset))])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            # transforms.Resize((14,14)) #TODO: change accordingly
                            transforms.Resize((28,28)),
                            transforms.Lambda(lambda x: torch.flatten(x))
                        ])),
    batch_size=batch_size, shuffle=True)

def train(model, train_loader, optimizer, criterion, epochs=10, val_loader=None, verbose=True):
    model.train()

    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)   
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            # print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
        train_acc = 100. * correct / len(train_loader.dataset)
        train_accs.append(train_acc)

        if val_loader is not None:
            print("Validation: ", end = "")
            ta = test(model, val_loader, criterion)
            test_accs.append(ta)

    return train_accs, test_accs

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

 # for saving activations

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
# manual save of activation
for i in range(len(model)):
    model[i].register_forward_hook(get_activation(str(i)))

train(model, train_loader, optimizer, criterion, epochs=1, val_loader=val_loader)

def run_exp(model, batch_size, epsilon, growth_epoch, end_epoch, total_epoch, learning_rate, criterion, growth):
    train_accs= []
    test_accs = []
    epochs = np.arange(total_epoch)

    if growth:
        # with growth
        modded_model_grow = copy.deepcopy(model)
        modded_optimizer_grow = torch.optim.SGD(modded_model_grow.parameters(), lr=learning_rate)
        modded_optimizer_grow.load_state_dict(optimizer.state_dict())
        initial_scores = []

        for iter in range(growth_epoch):
            for i in range(len(modded_model_grow)-1):
                # print("The size of activation of layer {}: {}".format(i, modded_model_grow.activations[str(i)].shape))
                # print("The size of my activation of layer {}: {}".format(i, activation[str(i)].shape))
                max_rank = modded_model_grow[i].width()
                # score = NORTH_score(modded_model_grow.activations[str(i)], batchsize=batch_size)
                score = NORTH_score(activation[str(i)], batchsize=batch_size, threshold=epsilon)
                # score = NORTH_score(modded_model_grow[i].weight, batchsize=batch_size)
                if iter == 0:
                    initial_scores.append(score)
                initScore = 0.97 * initial_scores[i]
                to_add = max(0, int(modded_model_grow[i].weight.size()[0] * (score - initScore)))
                print("Layer {} score: {}/{}, neurons to add: {}".format(i, score, max_rank, to_add))

                # "iterative_orthogonalization" and "kaiming_uniform" are the two options in function north_select, "autoinit" not working
                modded_model_grow.grow(i, to_add, fanin_weights="kaiming_uniform", 
                                    optimizer=modded_optimizer_grow)
            
            print("The grown model now has {} effective parameters.".format(modded_model_grow.parameter_count(masked = False)))
            print("Validation after growing: ", end = "")
            test(modded_model_grow, val_loader, criterion)
            train_acc,test_acc = train(modded_model_grow, train_loader, modded_optimizer_grow, criterion, epochs=1, val_loader=val_loader, verbose=False)
            train_accs += train_acc
            test_accs += test_acc
        train_acc,test_acc = train(modded_model_grow, train_loader, modded_optimizer_grow, criterion, epochs=end_epoch, val_loader=val_loader)
        train_accs += train_acc
        test_accs += test_acc

        for j in range(len(modded_model_grow)):
            print("Layer {} weight matrix after growth {}".format(j, modded_model_grow[j].weight.size()))
        print("The grown model now has {} effective parameters.".format(modded_model_grow.parameter_count(masked = False)))
        test(modded_model_grow, val_loader, criterion)

        print("---------GROWTH-------------")
        print('Train accs: ', train_accs)
        print('Test accs: ', test_accs)
        print("The model now has {} effective parameters.".format(model.parameter_count(masked = False)))
        plt.plot(epochs, train_accs, label='Training Accuracy', color='blue')
        plt.plot(epochs, test_accs, label='Testing Accuracy', color='purple')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.ylim(0, 100)
        plt.legend()
        plt.show()
        # train final
        train_accs= []
        test_accs = []
        for i in range(len(modded_model_grow)):
            model[i].reset_parameters()
        train_acc,test_acc = train(modded_model_grow, train_loader, modded_optimizer_grow, criterion, epochs=total_epoch, val_loader=val_loader)
        train_accs += train_acc
        test_accs += test_acc

        print("--------FINAL--------------")
        print('Train accs: ', train_accs)
        print('Test accs: ', test_accs)
        print("The model now has {} effective parameters.".format(model.parameter_count(masked = False)))
        plt.plot(epochs, train_accs, label='Training Accuracy', color='blue')
        plt.plot(epochs, test_accs, label='Testing Accuracy', color='purple')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.ylim(0, 100)
        plt.legend()
        plt.show()
        for j in range(len(modded_model_grow)):
            print("Layer {} weight matrix after growth {}".format(j, modded_model_grow[j].weight.size()))
        print("The grown model now has {} effective parameters.".format(modded_model_grow.parameter_count(masked = False)))
        test(modded_model_grow, val_loader, criterion)
    else:
        modded_model_static = copy.deepcopy(model)
        modded_optimizer_static = torch.optim.SGD(modded_model_static.parameters(), lr=learning_rate)
        modded_optimizer_static.load_state_dict(optimizer.state_dict())
        train_acc,test_acc = train(modded_model_static, train_loader, modded_optimizer_static, criterion, epochs=total_epoch, val_loader=val_loader)
        train_accs += train_acc
        test_accs += test_acc

        print("----------------------")
        print('Train accs: ', train_accs)
        print('Test accs: ', test_accs)
        print("The model now has {} effective parameters.".format(model.parameter_count(masked = False)))
        plt.plot(epochs, train_accs, label='Training Accuracy', color='blue')
        plt.plot(epochs, test_accs, label='Testing Accuracy', color='purple')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.ylim(0, 100)
        plt.legend()
        plt.show()

    return train_accs, test_accs

run_exp(model, batch_size, epsilon, growth_epoch, end_epoch, total_epoch, learning_rate, criterion, True)
# run_exp(model, batch_size, epsilon, growth_epoch, end_epoch, total_epoch, learning_rate, criterion, False)

