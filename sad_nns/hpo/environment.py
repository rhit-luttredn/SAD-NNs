from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchrl.envs
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
)
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp


class HPOEnv(torchrl.envs.EnvBase):
    def __init__(self, device="cpu", batch_size=[], seed=None):
        self._max_epoch = 255
        self._X_features_range = (10, 200)
        self._Y_features_range = (1, 20)
        self._samples_range = (50, 2000)
        super().__init__(device=device, batch_size=batch_size)
        self.loss_fn = nn.MSELoss()
        self._make_spec()
        if seed is not None:
            self._set_seed(seed)

    def _set_seed(self, seed: Optional[int]):
        np.random.seed(seed)
        # Possibly swap this and the dataset generator to use torch

    def reward_fn(self, loss):
        return (0.5 ** (loss / self.std)) * (0.95 ** (self.epoch / self.max_epoch))

    def _make_spec(self):
        self.observation_spec = CompositeSpec(
            # Loss Features
            train_loss=UnboundedContinuousTensorSpec(
                device=self.device, dtype=torch.float32, shape=()
            ),
            valid_loss=UnboundedContinuousTensorSpec(
                device=self.device, dtype=torch.float32, shape=()
            ),
            prev_train_loss=UnboundedContinuousTensorSpec(
                device=self.device, dtype=torch.float32, shape=()
            ),
            prev_valid_loss=UnboundedContinuousTensorSpec(
                device=self.device, dtype=torch.float32, shape=()
            ),
            # Budget Features
            max_epoch=BoundedTensorSpec(
                low=0, high=self._max_epoch, dtype=torch.int64, shape=()
            ),
            epoch=BoundedTensorSpec(
                low=0, high=self._max_epoch, dtype=torch.int64, shape=()
            ),
            # Dataset Meta-Features
            X_features=BoundedTensorSpec(
                low=self._X_features_range[0],
                high=self._X_features_range[1],
                dtype=torch.int64,
                shape=(),
            ),
            Y_features=BoundedTensorSpec(
                low=self._Y_features_range[0],
                high=self._Y_features_range[1],
                dtype=torch.int64,
                shape=(),
            ),
            std=UnboundedContinuousTensorSpec(
                device=self.device, dtype=torch.float32, shape=()
            ),
            train_samples=BoundedTensorSpec(
                low=self._samples_range[0],
                high=self._samples_range[1],
                dtype=torch.int64,
                shape=(),
            ),
            valid_samples=BoundedTensorSpec(
                low=self._samples_range[0],
                high=self._samples_range[1],
                dtype=torch.int64,
                shape=(),
            ),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        # self.action_spec = CompositeSpec(
        #     stop = BinaryDiscreteTensorSpec(1,device=self.device,dtype=torch.bool, shape = ()),
        #     lr = BoundedTensorSpec(low=-1,high=1,dtype=torch.float32, shape = ()),
        #     batch_size = BoundedTensorSpec(low=0,high=1,dtype=torch.float32, shape = ()),
        #     shape = (),
        # )
        self.action_spec = BoundedTensorSpec(
            low=torch.Tensor([0, -0.001, 0]),
            high=torch.Tensor([1, 0.001, 1]),
            shape=(*self.batch_size, 3),
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*self.batch_size, 1))

    def set_learning_rate(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _step(self, tensordict):
        # stop = tensordict['stop']
        stop = tensordict["action"][0] > 0.5
        # lr = tensordict['lr']
        lr = tensordict["action"][1]
        # batch_size = tensordict['batch_size']
        batch_size = tensordict["action"][2]
        batch_size = max(batch_size, 1.0 / self.dataset_shape[3])
        self.epoch += 1
        was_enabled = torch.is_grad_enabled()
        self.set_learning_rate(lr)
        torch.set_grad_enabled(True)
        self.model.train()

        # Call a provided `train()` hook
        # for frac in np.arange(0, 1, batch_size):
        #     i0 = int(frac * len(self.X_train))
        #     i1 = int(min((frac + batch_size) * len(self.X_train), len(self.X_train)))
        #     X_batch = self.X_train[i0:i1]
        #     Y_batch = self.Y_train[i0:i1]
        #     Y_pred = self.model(X_batch)
        #     loss = self.loss_fn(Y_pred, Y_batch)
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()

        torch.set_grad_enabled(was_enabled)
        self.model.eval()
        Y_pred = self.model(self.X_train)
        train_loss = self.loss_fn(Y_pred, self.Y_train).detach()
        Y_pred = self.model(self.X_valid)
        valid_loss = self.loss_fn(Y_pred, self.Y_valid).detach()
        raw_reward = self.reward_fn(valid_loss).view(*tensordict.shape, 1)
        reward = raw_reward - self.prev_reward
        self.prev_reward = raw_reward
        done = torch.zeros_like(reward, dtype=torch.bool)
        if stop or self.epoch >= self.max_epoch:
            Y_pred = self.model(self.X_test)
            test_loss = self.loss_fn(Y_pred, self.Y_test).detach()
            reward += self.reward_fn(test_loss).view(*tensordict.shape, 1)
            done = torch.ones_like(reward, dtype=torch.bool)
        troublemakers = [train_loss.numpy(), valid_loss.numpy()]
        if np.isnan(troublemakers).any() or np.isinf(troublemakers).any():
            done = torch.ones_like(reward, dtype=torch.bool)
            out = TensorDict(
                {
                    "train_loss": 0,
                    "valid_loss": 0,
                    "prev_train_loss": self.prev_train_loss.mean(),
                    "prev_valid_loss": self.prev_valid_loss.mean(),
                    "max_epoch": self.max_epoch,
                    "epoch": self.epoch,
                    "X_features": self.dataset_shape[0],
                    "Y_features": self.dataset_shape[1],
                    "train_samples": self.dataset_shape[2],
                    "valid_samples": self.dataset_shape[3],
                    "std": self.std,
                    "reward": torch.Tensor([-1]).view(*tensordict.shape, 1),
                    "done": done,
                },
                batch_size=tensordict.shape,
            )
            return out
        out = TensorDict(
            {
                "train_loss": train_loss.mean(),
                "valid_loss": valid_loss.mean(),
                "prev_train_loss": self.prev_train_loss.mean(),
                "prev_valid_loss": self.prev_valid_loss.mean(),
                "max_epoch": self.max_epoch,
                "epoch": self.epoch,
                "X_features": self.dataset_shape[0],
                "Y_features": self.dataset_shape[1],
                "train_samples": self.dataset_shape[2],
                "valid_samples": self.dataset_shape[3],
                "std": self.std,
                "reward": reward,
                "done": done,
            },
            batch_size=tensordict.shape,
        )
        self.prev_train_loss = train_loss
        self.prev_valid_loss = valid_loss
        return out

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            tensordict = TensorDict({}, [])
            if self.batch_size:
                tensordict = tensordict.expand(self.batch_size).contiguous()
        self.dataset = random_regression()
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = self.dataset
        self.dataset_shape = (
            X_train.shape[1],
            Y_train.shape[1],
            Y_train.shape[0],
            Y_valid.shape[0],
            Y_test.shape[0],
        )
        self.X_train = torch.Tensor(X_train)
        self.Y_train = torch.Tensor(Y_train)
        self.std = float(Y_train.std())
        self.X_valid = torch.Tensor(X_valid)
        self.Y_valid = torch.Tensor(Y_valid)
        self.X_test = torch.Tensor(X_test)
        self.Y_test = torch.Tensor(Y_test)
        self.model = nn.Sequential(
            nn.Linear(self.X_train.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.Y_train.shape[1]),
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.model.eval()
        train_loss = self.loss_fn(self.model(self.X_train), self.Y_train).detach()
        self.prev_train_loss = train_loss
        valid_loss = self.loss_fn(self.model(self.X_valid), self.Y_valid).detach()
        self.prev_valid_loss = valid_loss
        self.epoch = 0
        self.max_epoch = int(self._max_epoch * (np.random.random() * 0.5 + 0.5))
        self.prev_reward = self.reward_fn(valid_loss)
        out = TensorDict(
            {
                "train_loss": train_loss.mean(),
                "valid_loss": valid_loss.mean(),
                "prev_train_loss": self.prev_train_loss.mean(),
                "prev_valid_loss": self.prev_valid_loss.mean(),
                "max_epoch": self.max_epoch,
                "epoch": self.epoch,
                "X_features": self.dataset_shape[0],
                "Y_features": self.dataset_shape[1],
                "train_samples": self.dataset_shape[2],
                "std": self.std,
                "valid_samples": self.dataset_shape[3],
            },
            batch_size=tensordict.shape,
        )
        return out


def random_regression(
    param_mean=[500, 100, 0.1, 5, 0, 0, 0.5, 0, 0.3, 0.3],
    param_var=[100, 10, 0.01, 2, 10, 3, 0.2, 1, 0.1, 0.1],
):
    random_params = np.clip(
        np.random.normal(param_mean, param_var),
        [50, 10, 0, 1, -999, -10, 0, 0, 0.1, 0.1],
        [2000, 200, 1, 20, 999, 10, 1, 5, 0.5, 0.5],
    )
    X, Y = make_regression(
        int(random_params[0]),
        int(random_params[1]),
        n_informative=int(random_params[1] * random_params[2]),
        n_targets=int(random_params[3]),
        bias=random_params[4],
        effective_rank=None,  # if random_params[5]<0 else int(random_params[5]), this seems to have been causing errors
        tail_strength=random_params[6],
        noise=random_params[7],
    )
    if np.isnan(X).any() or np.isnan(Y).any() or np.isinf(X).any() or np.isinf(Y).any():
        return random_regression(param_mean, param_var)
    X = np.reshape(X, (X.shape[0], -1))
    Y = np.reshape(Y, (Y.shape[0], -1))
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=random_params[8]
    )
    X_valid, X_test, Y_valid, Y_test = train_test_split(
        X_temp, Y_temp, test_size=random_params[9]
    )
    return (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
