from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

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
      return random_regression(param_mean,param_var)

    X = np.reshape(X,(X.shape[0],-1))
    Y = np.reshape(Y,(Y.shape[0],-1))

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=random_params[8]
    )

    X_valid, X_test, Y_valid, Y_test = train_test_split(
        X_temp, Y_temp, test_size=random_params[9]
    )

    return (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

import torch.nn as nn
import torchrl.envs

from collections import defaultdict
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import tqdm
# from tensordict.nn import TensorDictModule
# from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn

from gymnasium import spaces
from gymnasium.envs.registration import register
from torchrl.envs import (
    CatTensors,
    GymEnv,
    GymWrapper,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp


class DemoEnv(gym.Env):
  def __init__(self, device="cpu", seed=None):
    # TODO: torchrl Environments can take device and batch_size, do I need to do anything special for that?
    self.device = device
    if seed is not None:
      self._set_seed(seed)

    self._max_epoch = 255
    self._X_features_range = np.array([10, 200])
    self._Y_features_range = np.array([1, 20])
    self._samples_range = np.array([50, 2000])

    self.epoch = 0
    self.max_epoch = int(self._max_epoch * (np.random.random() * 0.5 + 0.5))

    self.loss_fn = nn.MSELoss()
    self._make_space()

  def _set_seed(self, seed: Optional[int]):
    # TODO: Possibly swap this and the dataset generator to use torch
    # TODO: Does this even work?
    self.seed = seed
    np.random.seed(seed)

    if self.device == "cpu":
      torch.manual_seed(seed)
    else:
      torch.cuda.manual_seed(seed)

  def reward_fn(self, loss):
    loss = loss - self.prev_valid_loss
    loss = loss/self.Y_std
    if loss > 0:
      reward = 0.5**loss - 1
    else:
      reward = -loss
    # reward = (0.5**loss)
    # reward = -loss
    return reward.item()

  def _make_space(self):
    # TODO: Make sure I refactored this correctly.
    self.observation_space = spaces.Dict({
        # Loss Features
        "train_loss": spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
        "valid_loss": spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
        "prev_train_loss": spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
        "prev_valid_loss": spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
        # Budget Features
        "max_epoch": spaces.Box(low=0, high=self._max_epoch, shape=(), dtype=np.int64),
        "epoch": spaces.Box(low=0, high=self._max_epoch, shape=(), dtype=np.int64),
        # Dataset Meta-Features
        "X_features": spaces.Box(low=self._X_features_range[0], high=self._X_features_range[1], shape=(), dtype=np.int64),
        "Y_features": spaces.Box(low=self._Y_features_range[0], high=self._Y_features_range[1], shape=(), dtype=np.int64),
        "train_samples": spaces.Box(low=self._samples_range[0], high=self._samples_range[1], shape=(), dtype=np.int64),
        "valid_samples": spaces.Box(low=self._samples_range[0], high=self._samples_range[1], shape=(), dtype=np.int64),
    })
    # self.observation_space = spaces.Tuple((
    #     # Loss Features
    #     spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
    #     spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
    #     spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
    #     spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(), dtype=np.float32),
    #     # Budget Features
    #     spaces.Box(low=0, high=self._max_epoch, shape=(), dtype=np.int64),
    #     spaces.Box(low=0, high=self._max_epoch, shape=(), dtype=np.int64),
    #     # Dataset Meta-Features
    #     spaces.Box(low=self._X_features_range[0], high=self._X_features_range[1], shape=(), dtype=np.int64),
    #     spaces.Box(low=self._Y_features_range[0], high=self._Y_features_range[1], shape=(), dtype=np.int64),
    #     spaces.Box(low=self._samples_range[0], high=self._samples_range[1], shape=(), dtype=np.int64),
    #     spaces.Box(low=self._samples_range[0], high=self._samples_range[1], shape=(), dtype=np.int64),
    # ))

    # TODO: Look this over and make sure I refactored this correctly.
    # TODO: Will turning this into a Dict cause problems with torchrl's GymEnv?
    self.action_space = spaces.Box(
        # "stop": spaces.Discrete(2),
        # "learning_rate": spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
        # "exp_scale": spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
        # "batch_size": spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
        low = np.array([0.0, -1.0, 0.0, 0.0]),
        high = np.array([1.0, 1.0, 100.0, 1.0]),
        dtype = np.float32,
        shape = (4,),
    )

  def set_learning_rate(self, lr):
    for g in self.optimizer.param_groups:
      g['lr'] = lr

  def _get_obs(self, train_loss, valid_loss):
    return {
        "train_loss": train_loss/self.Y_std,
        "valid_loss": valid_loss/self.Y_std,
        "prev_train_loss": self.prev_train_loss.mean(),
        "prev_valid_loss": self.prev_valid_loss.mean(),
        "max_epoch": self.max_epoch,
        "epoch": self.epoch,
        "X_features": self.dataset_shape[0],
        "Y_features": self.dataset_shape[1],
        "train_samples": self.dataset_shape[2],
        "valid_samples": self.dataset_shape[3],
    }
    # return (
    #     train_loss,
    #     valid_loss,
    #     self.prev_train_loss.mean(),
    #     self.prev_valid_loss.mean(),
    #     self.max_epoch,
    #     self.epoch,
    #     self.dataset_shape[0],
    #     self.dataset_shape[1],
    #     self.dataset_shape[2],
    #     self.dataset_shape[3],
    # )

  def _get_info(self):
    return {}

  def step(self, action):
    # stop = bool(action['stop'])
    # stop = action[0] > 0.5
    stop = False

    # lr = action['learning_rate'] * (0.5**action['exp_scale'])
    lr = action[1] * (0.5**action[2])
    self.set_learning_rate(lr)

    # TODO: Make sure i refactored this right
    # Calculate batch size and shuffle data
    training_samples = self.dataset_shape[3]
    # batch_size = int(action['batch_size'] * training_samples)
    batch_size = int(action[3] * training_samples)
    batch_size = max(batch_size, 1)
    perm = np.random.permutation(training_samples)

    # Set up new training cycle
    self.epoch += 1;
    was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    self.model.train()

    # Train the model
    # TODO: Make sure I refactored this right
    for i in range(0, training_samples, batch_size):
      batch = perm[i:i+batch_size]
      X_batch = self.X_train[batch]
      Y_batch = self.Y_train[batch]
      Y_pred = self.model(X_batch)

      loss = self.loss_fn(Y_pred, Y_batch)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    torch.set_grad_enabled(was_enabled)

    # Validate the model
    self.model.eval()

    Y_pred = self.model(self.X_train)
    train_loss = self.loss_fn(Y_pred, self.Y_train).detach()

    Y_pred = self.model(self.X_valid)
    valid_loss = self.loss_fn(Y_pred, self.Y_valid).detach()

    # Calculate the reward
    reward = self.reward_fn(valid_loss)
    # reward = raw_reward - self.prev_reward
    # self.prev_reward = raw_reward

    # Determine if we should stop traing
    done = False
    if stop or self.epoch >= self.max_epoch:
      Y_pred = self.model(self.X_test)
      test_loss = self.loss_fn(Y_pred, self.Y_test).detach()
      reward += self.reward_fn(test_loss)
      done = True

    # TODO: Is this comment right?
    # If we end up with any nan's or inf's in our loss, stop training and give reward -1
    troublemakers = [train_loss.detach().numpy(), valid_loss.detach().numpy()]
    if np.isnan(troublemakers).any() or np.isinf(troublemakers).any():
      observation = self._get_obs(0, 0)
      reward = -1.0
      done = True
      info = self._get_info()
      return observation, reward, done, False, info

    # Finish step normally
    observation = self._get_obs(train_loss.mean(), valid_loss.mean())
    reward = reward
    done = done
    info = self._get_info()

    # Previous info to be the current info
    # TODO: Should this be .mean()?
    self.prev_train_loss = train_loss
    self.prev_valid_loss = valid_loss
    return observation, reward, done, False, info

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # Generate a random regression dataset
    self.dataset = random_regression()
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = self.dataset
    self.dataset_shape = (X_train.shape[1], Y_train.shape[1], Y_train.shape[0], Y_valid.shape[0], Y_test.shape[0])
    self.Y_std = np.std(np.concatenate((Y_train,Y_valid),axis=0)).mean()

    # TODO: Is this needed? Pretty sure we can give torch models np arrays. Might be good for gpu???
    self.X_train = torch.Tensor(X_train)
    self.Y_train = torch.Tensor(Y_train)
    self.X_valid = torch.Tensor(X_valid)
    self.Y_valid = torch.Tensor(Y_valid)
    self.X_test = torch.Tensor(X_test)
    self.Y_test = torch.Tensor(Y_test)

    # Generate a model for the dataset
    self.model = nn.Sequential(
      nn.Linear(self.X_train.shape[1], 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.Y_train.shape[1]),
    )
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.001)

    # Evaluate the initial model
    self.model.eval()
    train_loss = self.loss_fn(self.model(self.X_train), self.Y_train).detach()
    valid_loss = self.loss_fn(self.model(self.X_valid), self.Y_valid).detach()

    # TODO: Should this be .mean()??
    self.prev_train_loss = train_loss
    self.prev_valid_loss = valid_loss
    self.prev_reward = self.reward_fn(valid_loss)

    # Initialize some other values
    self.epoch = 0
    self.max_epoch = int(self._max_epoch * (np.random.random() * 0.5 + 0.5))

    # Get the return values
    observation = self._get_obs(train_loss.mean(), valid_loss.mean())
    info = self._get_info()
    return observation, info

register(
    id='DemoEnv-v1',
    entry_point=__name__ + ':DemoEnv'
)

class BaseEnv(gym.Env):
    def __init__(self, device="cpu", seed=None):
        # TODO: torchrl Environments can take device and batch_size, do I need to do anything special for that?
        self.device = device
        if seed is not None:
          self._set_seed(seed)
    
        self.loss_fn = nn.MSELoss()
        self._make_space()

    def _make_space(self):
        self.observation_space = spaces.Dict({
            #TODO
        })
        self.action_space = spaces.Box(
            #TODO
        )