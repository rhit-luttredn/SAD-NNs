from copy import copy

import torch.nn as nn
from gymnasium.vector import VectorEnv
from sad_nns.utils import generate_feature_extractor, layer_init
from torch.distributions.categorical import Categorical
from torchinfo import summary


class PPOAgent(nn.Module):
    def __init__(
            self,
            envs: VectorEnv, 
            conv_sizes: list = [16, 32, 64],
            kernel_sizes: list|int = 2,
            linear_sizes: list = [],
            out_features: int = 512
    ):
        super().__init__()
        if len(conv_sizes) == 0:
            raise ValueError("There must be at least one convolutional layer")
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_sizes)

        if len(kernel_sizes) != len(conv_sizes):
            raise ValueError("The number of kernel sizes must be one less than the number of convolutional layers")

        conv_sizes = copy(conv_sizes)
        kernel_sizes = copy(kernel_sizes)
        linear_sizes = list(copy(linear_sizes))
        linear_sizes.append(out_features)

        self.height = envs.single_observation_space.shape[0]
        self.width = envs.single_observation_space.shape[1]
        self.input_channels = envs.single_observation_space.shape[2]

        if any([x > self.height for x in kernel_sizes]):
            raise ValueError("The kernel size cannot be larger than the input size")

        input_shape = envs.single_observation_space.shape
        self.network = generate_feature_extractor(input_shape, conv_sizes, kernel_sizes, linear_sizes)
        # self.network_summary = summary(self.network, input_size=(1, self.input_channels, self.height, self.width), verbose=0)

        self.actor = layer_init(nn.Linear(out_features, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(out_features, 1), std=1)

    def get_value(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        x = x.permute(0, 3, 1, 2)
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)