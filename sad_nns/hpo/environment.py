import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.envs.registration import register
from sad_nns.utils import random_regression


class RegressionEnv(gym.Env):
    _max_epoch = 1024
    _min_epoch = 128
    _X_features_range = (10, 200)
    _Y_features_range = (1, 20)
    _samples_range = (500, 2000)
    _layers_range = (0, 10)
    _size_range = (0, 200000)
    _nodes_range = (8, 128)
    reward_beta = 0.1  # TODO: Use this to adjust reward smoothing

    def __init__(
        self,
        device="cpu",
        seed=None,
        action_low=[0.0, -1.0, 0.0],
        action_high=[1.0, 1.0, 1.0],
    ):
        self._action_low = action_low
        self._action_high = action_high
        self.device = device
        if seed is not None:
            self._set_seed(seed)
        self._make_space()
        self.loss_fn = nn.MSELoss()

    def _set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

        if self.device == "cpu":
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)

    def reward_fn(self, loss, model_size, expended_budget):
        loss_factor = 1 - loss / self.loss_baseline
        size_factor = min(1, max(0, 1 - model_size / self.max_size))
        budget_factor = min(1, max(0, 1 - expended_budget / self.budget))
        if loss_factor < 0:  # safety net to avoid complex results
            loss_factor = np.tanh(
                loss_factor
            )  # activate negative losses to give a lower bound on reward
            return (
                -(abs(loss_factor) ** self.loss_importance)
                * (size_factor**self.size_importance)
                * (budget_factor**self.budget_importance)
            )
        return (
            (loss_factor**self.loss_importance)
            * (size_factor**self.size_importance)
            * (budget_factor**self.budget_importance)
        )

    def _make_space(self):
        self.observation_space = spaces.Dict(
            {
                # Loss Features
                "train_loss": spaces.Box(
                    low=np.float32(0),
                    high=np.float32(np.inf),
                    shape=(),
                    dtype=np.float32,
                ),
                "valid_loss": spaces.Box(
                    low=np.float32(0),
                    high=np.float32(np.inf),
                    shape=(),
                    dtype=np.float32,
                ),
                "baseline_loss": spaces.Box(
                    low=np.float32(0),
                    high=np.float32(np.inf),
                    shape=(),
                    dtype=np.float32,
                ),
                # Budget Features
                "max_epoch": spaces.Box(
                    low=0, high=self._max_epoch, shape=(), dtype=np.int64
                ),
                "epoch": spaces.Box(
                    low=0, high=self._max_epoch, shape=(), dtype=np.int64
                ),
                # Dataset Meta-Features
                "X_features": spaces.Box(
                    low=self._X_features_range[0],
                    high=self._X_features_range[1],
                    shape=(),
                    dtype=np.int64,
                ),
                "Y_features": spaces.Box(
                    low=self._Y_features_range[0],
                    high=self._Y_features_range[1],
                    shape=(),
                    dtype=np.int64,
                ),
                "train_samples": spaces.Box(
                    low=0, high=self._samples_range[1], shape=(), dtype=np.int64
                ),
                "valid_samples": spaces.Box(
                    low=0, high=self._samples_range[1], shape=(), dtype=np.int64
                ),
                # Base Model Features
                "layers": spaces.Box(
                    low=self._layers_range[0],
                    high=self._layers_range[1],
                    shape=(),
                    dtype=np.int64,
                ),
                "model_size": spaces.Box(
                    low=self._size_range[0],
                    high=self._size_range[1],
                    shape=(),
                    dtype=np.int64,
                ),
                "max_size": spaces.Box(
                    low=self._size_range[0],
                    high=self._size_range[1] * 2,
                    shape=(),
                    dtype=np.int64,
                ),
                # Reward features
                "loss_importance": spaces.Box(
                    low=np.float32(0), high=np.float32(1), shape=(), dtype=np.float32
                ),
                "size_importance": spaces.Box(
                    low=np.float32(0), high=np.float32(1), shape=(), dtype=np.float32
                ),
                "budget_importance": spaces.Box(
                    low=np.float32(0), high=np.float32(1), shape=(), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Box(
            low=np.array(self._action_low),
            high=np.array(self._action_high),
            dtype=np.float32,
            shape=(3,),
        )

    def _get_observation(self, train_loss, valid_loss):
        return {
            # Loss Features
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "baseline_loss": self.loss_baseline,
            # Budget Features
            "max_epoch": self.max_epoch,
            "epoch": self.epoch,
            # Dataset Meta-Features
            "X_features": self.dataset_shape[0],
            "Y_features": self.dataset_shape[1],
            "train_samples": self.dataset_shape[2],
            "valid_samples": self.dataset_shape[3],
            # Base Model Features
            "layers": len(
                [
                    module
                    for module in self.model.modules()
                    if isinstance(module, nn.Linear)
                ]
            ),
            "model_size": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "max_size": self.max_size,
            # Reward features
            "loss_importance": self.loss_importance,
            "size_importance": self.size_importance,
            "budget_importance": self.budget_importance,
        }

    def _get_info(self):
        return {}

    def _set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def reset(self, seed=None, options=None):
        super().reset()
        if seed is not None:
            self._set_seed(seed)
        self.dataset = random_regression()
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = self.dataset
        self.dataset_shape = (
            X_train.shape[1],
            Y_train.shape[1],
            Y_train.shape[0],
            Y_valid.shape[0],
            Y_test.shape[0],
        )
        self.loss_baseline = np.std(np.concatenate((Y_train, Y_valid), axis=0)) ** 2

        self.X_train = torch.Tensor(X_train)
        self.Y_train = torch.Tensor(Y_train)
        self.X_valid = torch.Tensor(X_valid)
        self.Y_valid = torch.Tensor(Y_valid)
        self.X_test = torch.Tensor(X_test)
        self.Y_test = torch.Tensor(Y_test)

        self.model = nn.Sequential()

        num_layers = np.random.randint(self._layers_range[0], self._layers_range[1])
        num_nodes = np.random.randint(self._nodes_range[0], self._nodes_range[1])

        in_features = self.dataset_shape[0]
        for i in range(num_layers):
            out_features = num_nodes
            self.model.append(nn.Linear(in_features, out_features))
            self.model.append(nn.ReLU())
            in_features = out_features
        out_features = self.dataset_shape[1]
        self.model.append(nn.Linear(in_features, out_features))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0, betas=(0, 0))
        # setting betas to zero gets us reasonably close to SGD
        # torch.optim.SGD just doesnt work

        self.max_size = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        ) * (1.5 + np.random.random())

        self.model.eval()
        train_loss = self.loss_fn(self.model(self.X_train), self.Y_train).mean().item()
        valid_loss = self.loss_fn(self.model(self.X_valid), self.Y_valid).mean().item()

        self.epoch = 0
        self.max_epoch = int(
            self._min_epoch + np.random.random() * (self._max_epoch - self._min_epoch)
        )
        self.budget = self.max_epoch
        self.loss_importance = 0.6 + np.random.random() * 0.2
        self.budget_importance = (1 - self.loss_importance) * np.random.random() / 4
        self.size_importance = 1 - self.loss_importance - self.budget_importance

        self.prev_reward = self.reward_fn(
            valid_loss,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            0,
        )
        self.initial_reward = self.prev_reward
        self.avg_reward = 0

        observation = self._get_observation(train_loss, valid_loss)
        info = self._get_info()

        # print(observation, info)
        return observation, info

    def step(self, action):
        stop = action[0] > 0.5
        lr = action[1]
        batch_ratio = action[2]

        self._set_lr(lr)

        training_samples = self.dataset_shape[3]
        batch_size = int(batch_ratio * training_samples)
        batch_size = max(batch_size, 1)
        perm = np.random.permutation(training_samples)

        self.epoch += 1
        was_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(
            True
        )  # Not really sure why but it won't work without this
        self.model.train()

        self.model.eval()
        Y_pred = self.model(self.X_train)
        train_loss = self.loss_fn(Y_pred, self.Y_train).mean().item()

        Y_pred = self.model(self.X_valid)
        valid_loss = self.loss_fn(Y_pred, self.Y_valid).mean().item()

        troublemakers = [train_loss, valid_loss]
        if np.isnan(troublemakers).any() or np.isinf(troublemakers).any():
            observation = self._get_observation(0, 0)
            reward = -1.5
            done = True
            info = self._get_info()
            # print(observation, reward, done, False, info)
            return observation, reward, done, False, info

        for i in range(0, training_samples, batch_size):
            batch = perm[i : i + batch_size]
            X_batch = self.X_train[batch]
            Y_batch = self.Y_train[batch]
            Y_pred = self.model(X_batch)

            loss = self.loss_fn(Y_pred, Y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        torch.set_grad_enabled(was_enabled)

        self.model.eval()
        Y_pred = self.model(self.X_train)
        train_loss = self.loss_fn(Y_pred, self.Y_train).mean().item()

        Y_pred = self.model(self.X_valid)
        valid_loss = self.loss_fn(Y_pred, self.Y_valid).mean().item()

        base_reward = self.reward_fn(
            valid_loss,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            self.epoch,
        )
        self.avg_reward = (base_reward - self.prev_reward) * self.reward_beta + (
            1 - self.reward_beta
        ) * self.avg_reward
        self.prev_reward = base_reward

        done = False
        if stop or self.epoch >= self.max_epoch:
            Y_pred = self.model(self.X_test)
            test_loss = self.loss_fn(Y_pred, self.Y_test).mean().item()
            self.avg_reward += (
                self.reward_fn(
                    test_loss,
                    sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    self.epoch,
                )
                * self.reward_beta
                + (1 - self.reward_beta) * self.avg_reward
            )
            done = True

        # handle any NANs or Infs which will ruin everything
        troublemakers = [train_loss, valid_loss]
        if np.isnan(troublemakers).any() or np.isinf(troublemakers).any():
            observation = self._get_observation(0, 0)
            reward = -1.5
            done = True
            info = self._get_info()
            # print(observation, reward, done, False, info)
            return observation, reward, done, False, info

        observation = self._get_observation(train_loss, valid_loss)
        done = done
        info = self._get_info()
        # print(observation, self.avg_reward, done, False, info)
        return observation, self.avg_reward, done, False, info


register(id="RegressionEnv-v0", entry_point=__name__ + ":RegressionEnv")
##########################################################################################

from external.NeurOps.pytorch.neurops import *


class NORTH_RegressionEnv(gym.Env):
    _max_epoch = 1024
    _min_epoch = 128
    _X_features_range = (10, 200)
    _Y_features_range = (1, 20)
    _samples_range = (500, 2000)
    _layers_range = (0, 10)
    _size_range = (0, 200000)
    _nodes_range = (8, 128)
    reward_beta = 0.1  # TODO: Use this to adjust reward smoothing

    ## action = [stop, learning_rate, batch_ratio, train, prune, grow]
    def __init__(
        self,
        device="cpu",
        seed=None,
        action_low=[0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        action_high=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ):
        self._action_low = action_low
        self._action_high = action_high
        self.device = device
        if seed is not None:
            self._set_seed(seed)
        self._make_space()
        self.loss_fn = nn.MSELoss()

    def _set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

        if self.device == "cpu":
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)

    def reward_fn(self, loss, model_size, expended_budget):
        loss_factor = 1 - loss / self.loss_baseline
        size_factor = min(1, max(0, 1 - model_size / self.max_size))
        budget_factor = min(1, max(0, 1 - expended_budget / self.budget))
        if loss_factor < 0:  # safety net to avoid complex results
            loss_factor = np.tanh(
                loss_factor
            )  # activate negative losses to give a lower bound on reward
            return (
                -(abs(loss_factor) ** self.loss_importance)
                * (size_factor**self.size_importance)
                * (budget_factor**self.budget_importance)
            )
        return (
            (loss_factor**self.loss_importance)
            * (size_factor**self.size_importance)
            * (budget_factor**self.budget_importance)
        )

    def _make_space(self):
        self.observation_space = spaces.Dict(
            {
                # Loss Features
                "train_loss": spaces.Box(
                    low=np.float32(0),
                    high=np.float32(np.inf),
                    shape=(),
                    dtype=np.float32,
                ),
                "valid_loss": spaces.Box(
                    low=np.float32(0),
                    high=np.float32(np.inf),
                    shape=(),
                    dtype=np.float32,
                ),
                "baseline_loss": spaces.Box(
                    low=np.float32(0),
                    high=np.float32(np.inf),
                    shape=(),
                    dtype=np.float32,
                ),
                # Budget Features
                "max_epoch": spaces.Box(
                    low=0, high=self._max_epoch, shape=(), dtype=np.int64
                ),
                "epoch": spaces.Box(
                    low=0, high=self._max_epoch, shape=(), dtype=np.int64
                ),
                # Dataset Meta-Features
                "X_features": spaces.Box(
                    low=self._X_features_range[0],
                    high=self._X_features_range[1],
                    shape=(),
                    dtype=np.int64,
                ),
                "Y_features": spaces.Box(
                    low=self._Y_features_range[0],
                    high=self._Y_features_range[1],
                    shape=(),
                    dtype=np.int64,
                ),
                "train_samples": spaces.Box(
                    low=0, high=self._samples_range[1], shape=(), dtype=np.int64
                ),
                "valid_samples": spaces.Box(
                    low=0, high=self._samples_range[1], shape=(), dtype=np.int64
                ),
                # Base Model Features
                "layers": spaces.Box(
                    low=self._layers_range[0],
                    high=self._layers_range[1],
                    shape=(),
                    dtype=np.int64,
                ),
                "model_size": spaces.Box(
                    low=self._size_range[0],
                    high=self._size_range[1],
                    shape=(),
                    dtype=np.int64,
                ),
                "max_size": spaces.Box(
                    low=self._size_range[0],
                    high=self._size_range[1] * 2,
                    shape=(),
                    dtype=np.int64,
                ),
                # Reward features
                "loss_importance": spaces.Box(
                    low=np.float32(0), high=np.float32(1), shape=(), dtype=np.float32
                ),
                "size_importance": spaces.Box(
                    low=np.float32(0), high=np.float32(1), shape=(), dtype=np.float32
                ),
                "budget_importance": spaces.Box(
                    low=np.float32(0), high=np.float32(1), shape=(), dtype=np.float32
                ),
                # Prune Stats
                "num_pruned": spaces.Box(
                    low=0, high=self._size_range[1] * 2, shape=(), dtype=np.int64
                ),
                # Grow Stats
                "num_grown": spaces.Box(
                    low=0, high=self._size_range[1] * 2, shape=(), dtype=np.int64
                ),
            }
        )

        self.action_space = spaces.Box(
            low=np.array(self._action_low),
            high=np.array(self._action_high),
            dtype=np.float32,
            shape=(6,),
        )

    def _get_observation(self, stats):
        return {
            # Loss Features
            "train_loss": stats["train_loss"],
            "valid_loss": stats["valid_loss"],
            "baseline_loss": self.loss_baseline,
            # Budget Features
            "max_epoch": self.max_epoch,
            "epoch": self.epoch,
            # Dataset Meta-Features
            "X_features": self.dataset_shape[0],
            "Y_features": self.dataset_shape[1],
            "train_samples": self.dataset_shape[2],
            "valid_samples": self.dataset_shape[3],
            # Base Model Features
            "layers": len(
                [
                    module
                    for module in self.model.modules()
                    if isinstance(module, nn.Linear)
                ]
            ),
            "model_size": self.model.parameter_count(masked=False),
            "max_size": self.max_size,
            # Reward features
            "loss_importance": self.loss_importance,
            "size_importance": self.size_importance,
            "budget_importance": self.budget_importance,
            # Prune Stats
            "num_pruned": stats["num_pruned"],
            # Grow Stats
            "num_grown": stats["num_grown"],
        }

    def _get_info(self):
        return {}

    def _set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def reset(self, seed=None, options=None):
        super().reset()
        if seed is not None:
            self._set_seed(seed)
        self.dataset = random_regression()
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = self.dataset
        self.dataset_shape = (
            X_train.shape[1],
            Y_train.shape[1],
            Y_train.shape[0],
            Y_valid.shape[0],
            Y_test.shape[0],
        )
        self.loss_baseline = np.std(np.concatenate((Y_train, Y_valid), axis=0)) ** 2

        self.X_train = torch.Tensor(X_train)  # .to(self.device)
        self.Y_train = torch.Tensor(Y_train)  # .to(self.device)
        self.X_valid = torch.Tensor(X_valid)  # .to(self.device)
        self.Y_valid = torch.Tensor(Y_valid)  # .to(self.device)
        self.X_test = torch.Tensor(X_test)  # .to(self.device)
        self.Y_test = torch.Tensor(Y_test)  # .to(self.device)

        num_layers = np.random.randint(self._layers_range[0], self._layers_range[1])
        num_nodes = np.random.randint(self._nodes_range[0], self._nodes_range[1])

        model_layers = nn.ModuleList()

        in_features = self.dataset_shape[0]
        for i in range(num_layers):
            out_features = num_nodes
            model_layers.append(
                ModLinear(
                    in_features,
                    out_features,
                )
            )  # masked=True, prebatchnorm=True, learnable_mask=True))
            in_features = out_features
        out_features = self.dataset_shape[1]
        model_layers.append(
            ModLinear(
                in_features,
                out_features,
                nonlinearity="",
            )
        )  # masked=True, prebatchnorm=True))
        model_layers.to(self.device)
        self.model = ModSequential(
            *model_layers,
            input_shape=(1, self.dataset_shape[0]),
            track_activations=False,
            track_auxiliary_gradients=True
        ).to(self.device)
        # torch.compile(self.model)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0
        )

        self.activation = {}

        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()

            return hook

        # manual save of activation
        for i in range(len(self.model) - 1):
            self.model[i].register_forward_hook(get_activation(str(i)))

        self.max_size = self.model.parameter_count(masked=False) * (
            1.5 + np.random.random()
        )

        self.model.eval()
        train_loss = self.loss_fn(self.model(self.X_train), self.Y_train).mean().item()
        valid_loss = self.loss_fn(self.model(self.X_valid), self.Y_valid).mean().item()

        self.epoch = 0
        self.max_epoch = int(
            self._min_epoch + np.random.random() * (self._max_epoch - self._min_epoch)
        )
        self.budget = self.max_epoch
        self.loss_importance = 0.6 + np.random.random() * 0.2
        self.budget_importance = (1 - self.loss_importance) * np.random.random() / 4
        self.size_importance = 1 - self.loss_importance - self.budget_importance
        self.initial_scores = None

        self.prev_reward = self.reward_fn(
            valid_loss, self.model.parameter_count(masked=False), 0
        )
        self.initial_reward = self.prev_reward
        self.avg_reward = 0
        stats = {
            "num_grown": 0,
            "num_pruned": 0,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }

        observation = self._get_observation(stats)
        info = self._get_info()

        # print(observation, info)
        return observation, info

    def train(self, batch_size):
        training_samples = self.dataset_shape[3]
        perm = np.random.permutation(training_samples)

        self.epoch += 1
        was_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(
            True
        )  # Not really sure why but it won't work without this
        self.model.train()

        for i in range(
            0, training_samples - batch_size, batch_size
        ):  # again making sure that we dont run a batch of size 1
            batch = perm[i : i + batch_size]
            X_batch = self.X_train[batch]
            Y_batch = self.Y_train[batch]
            Y_pred = self.model(X_batch)

            loss = self.loss_fn(Y_pred, Y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        torch.set_grad_enabled(was_enabled)

    def prune(self, stats):
        # TODO
        pass

    epsilon = 0.01

    def grow(self, batch_size, stats):
        is_first = False
        if self.initial_scores == None:
            self.initial_scores = []
            is_first = True
        for i in range(len(self.model) - 1):
            max_rank = self.model[i].width()
            score = NORTH_score(
                self.activation[str(i)], batchsize=batch_size, threshold=self.epsilon
            )  # TODO figure out how to parameterize
            # TODO figure out why i have squiggles
            if is_first:
                self.initial_scores.append(score)
            initScore = 0.97 * self.initial_scores[i]
            to_add = max(0, int(self.model[i].weight.size()[0] * (score - initScore)))

            self.model.grow(
                i, to_add, fanin_weights="kaiming_uniform", optimizer=self.optimizer
            )
        return stats

    def step(self, action):
        stop = action[0] > 0.5
        lr = action[1]
        batch_ratio = action[2]
        train = action[3]
        grow = action[4] > 0.5
        prune = action[5] > 0.5
        stats = {"num_grown": 0, "num_pruned": 0, "train_loss": 0, "valid_loss": 0}
        self._set_lr(lr)

        training_samples = self.dataset_shape[3]
        batch_size = int(batch_ratio * training_samples)
        batch_size = max(
            batch_size, 2
        )  # changed to 2 because there seems to be some batch norm

        self.train(batch_size)

        if prune:
            self.prune(stats)

        if grow:
            self.grow(batch_size, stats)

        self.model.eval()
        Y_pred = self.model(self.X_train)
        train_loss = self.loss_fn(Y_pred, self.Y_train).mean().item()

        Y_pred = self.model(self.X_valid)
        valid_loss = self.loss_fn(Y_pred, self.Y_valid).mean().item()

        base_reward = self.reward_fn(
            valid_loss, self.model.parameter_count(masked=False), self.epoch
        )
        self.avg_reward = (base_reward - self.prev_reward) * self.reward_beta + (
            1 - self.reward_beta
        ) * self.avg_reward
        self.prev_reward = base_reward

        done = False
        if stop or self.epoch >= self.max_epoch:
            Y_pred = self.model(self.X_test)
            test_loss = self.loss_fn(Y_pred, self.Y_test).mean().item()
            self.avg_reward += (
                self.reward_fn(
                    test_loss, self.model.parameter_count(masked=False), self.epoch
                )
                * self.reward_beta
                + (1 - self.reward_beta) * self.avg_reward
            )
            done = True

        stats["train_loss"] = train_loss
        stats["valid_loss"] = valid_loss

        # handle any NANs or Infs which will ruin everything
        troublemakers = [train_loss, valid_loss]
        if np.isnan(troublemakers).any() or np.isinf(troublemakers).any():
            stats["train_loss"] = 0
            stats["valid_loss"] = 0
            observation = self._get_observation(stats)
            reward = -1.5
            done = True
            info = self._get_info()
            # print(observation, reward, done, False, info)
            return observation, reward, done, False, info

        observation = self._get_observation(stats)
        done = done
        info = self._get_info()
        # print(observation, self.avg_reward, done, False, info)
        return observation, self.avg_reward, done, False, info


register(id="NORTH_RegressionEnv-v0", entry_point=__name__ + ":NORTH_RegressionEnv")
