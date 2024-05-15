from functools import reduce
from operator import mul

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from sad_nns.envs.wrappers import OneHotImageWrapper, FullyObsRotatingWrapper
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from torchinfo import summary


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def generate_feature_extractor(input_shape: tuple, conv_sizes: list, kernel_sizes: list, linear_sizes: list):
    print(conv_sizes, kernel_sizes, linear_sizes)
    height = input_shape[0]
    width = input_shape[1]
    input_channels = input_shape[2]

    # Create the convolutional layers
    conv_layers = []
    conv_layers.append(layer_init(nn.Conv2d(
        input_channels,
        conv_sizes[0],
        kernel_sizes[0],
        padding=int(0.5*kernel_sizes[0])
    )))
    conv_layers.append(nn.ReLU())
    for i in range(1, len(conv_sizes)):
        conv_layers.append(layer_init(nn.Conv2d(
            conv_sizes[i-1],
            conv_sizes[i],
            kernel_sizes[i],
            padding=int(0.5*kernel_sizes[0])
        )))
        conv_layers.append(nn.ReLU())

    conv_layers.append(nn.Flatten())
    conv_net = nn.Sequential(*conv_layers)

    # Calculate the size of the output of the conv_layers by doing one forward pass
    dummy_input = torch.randn(1, input_channels, height, width)
    output = conv_net(dummy_input)
    output_size = reduce(mul, output.shape, 1)

    # Create the linear layers
    linear_layers = []
    linear_layers.append(layer_init(nn.Linear(output_size, linear_sizes[0])))
    linear_layers.append(nn.ReLU())
    for i in range(1, len(linear_sizes)):
        linear_layers.append(layer_init(nn.Linear(linear_sizes[i-1], linear_sizes[i])))
        linear_layers.append(nn.ReLU())

    network = nn.Sequential(
        *conv_layers,
        *linear_layers,
    )
    return network


def random_regression(
    param_mean=[500, 100, 0.1, 5, 0, 0, 0.5, 0, 0.3, 0.3],
    param_var=[100, 10, 0.01, 2, 10, 3, 0.2, 1, 0.1, 0.1],
):
    random_params = np.clip(
        np.random.normal(param_mean, param_var),
        [500, 10, 0, 1, -999, -10, 0, 0, 0.1, 0.1],
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


def make_env(env_id, idx, capture_video, run_path, env_kwargs: dict = {}):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env = gym.wrappers.RecordVideo(env, f"../videos/{run_path}")
        else:
            env = gym.make(env_id, **env_kwargs)
        # env = FullyObsRotatingWrapper(env)
        env = OneHotImageWrapper(env)
        env = ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
