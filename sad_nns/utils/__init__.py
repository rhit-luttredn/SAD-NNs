from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
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