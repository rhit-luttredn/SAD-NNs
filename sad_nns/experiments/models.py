import torch.nn as nn
from neurops import *


class MNISTModel(ModSequential):
    def __init__(self):
        super().__init__(
            ModConv2d(in_channels=1, out_channels=8, kernel_size=7, masked=True, padding=1, learnable_mask=True),
            ModConv2d(in_channels=8, out_channels=16, kernel_size=7, masked=True, padding=1, prebatchnorm=True, learnable_mask=True),
            ModConv2d(in_channels=16, out_channels=16, kernel_size=5, masked=True, prebatchnorm=True, learnable_mask=True),
            ModLinear(16*18*18, 32, masked=True, prebatchnorm=True, learnable_mask=True),
            ModLinear(32, 10, masked=True, prebatchnorm=True, nonlinearity=""),
            track_activations=True,
            track_auxiliary_gradients=True,
            input_shape = (1, 28, 28)
        )
    

class CIFAR10Model(ModSequential):
    def __init__(self):
        super().__init__(
            ModConv2d(in_channels=3, out_channels=8, kernel_size=5, padding=1, masked=True, learnable_mask=True),
            ModConv2d(in_channels=8, out_channels=16, kernel_size=7, padding=1, postpool=nn.MaxPool2d(2,2), prebatchnorm=True, masked=True, learnable_mask=True),
            ModConv2d(in_channels=16, out_channels=16, kernel_size=7, postpool=nn.MaxPool2d(2,2), prebatchnorm=True, masked=True, learnable_mask=True),
            ModLinear(16*8*8, 128, prebatchnorm=True, preflatten=True, masked=True, learnable_mask=True),
            ModLinear(128, 32, masked=True, learnable_mask=True),
            ModLinear(32, 10, nonlinearity="", masked=True),
            track_activations=True,
            track_auxiliary_gradients=True,
            input_shape = (3, 32, 32)
        )
