import neurops
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv


class QNetwork(nn.Module):
    def __init__(self, envs: VectorEnv, linear_sizes: list = []):
        assert len(linear_sizes) > 0, "There must be at least one linear layer"
        super().__init__()
        height = envs.single_observation_space.shape[0]
        width = envs.single_observation_space.shape[1]
        n_input_channels = envs.single_observation_space.shape[2]

        conv_layers = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, 8, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )

        # Calculate the size of the output of the conv_layers by doing one forward pass
        dummy_input = torch.rand(1, n_input_channels, height, width)
        output = conv_layers(dummy_input)
        output_size = np.prod(output.shape)

        linear_sizes = [output_size] + list(linear_sizes) + [envs.single_action_space.n]
        linear_layers = []
        for i in range(len(linear_sizes) - 1):
            nonlinearity = "relu" if i < len(linear_sizes) - 2 else ""
            linear_layers.append(neurops.ModLinear(linear_sizes[i], linear_sizes[i + 1], 
                                                   predropout=True, nonlinearity=nonlinearity))

        self.growth_net = neurops.ModSequential(
            *linear_layers,
            track_activations=True,
            track_auxiliary_gradients=True,
            input_shape = (output_size)
        )

        # for saving activations
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook

        # manual save of activation
        for i in range(len(self.growth_net)):
            self.growth_net[i].register_forward_hook(get_activation(str(i)))

        self.network = nn.Sequential(
            conv_layers,
            nn.Flatten(),
            self.growth_net,
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.network(x / 255.0)

    def enable_dropout(self):
        for layer in self.growth_net:
            if isinstance(layer, neurops.ModLinear):
                layer.predropout.train()

    def disable_dropout(self):
        for layer in self.growth_net:
            if isinstance(layer, neurops.ModLinear):
                layer.predropout.eval()
    
    def get_linear_sizes(self):
        return [layer.out_features for layer in self.growth_net if isinstance(layer, neurops.ModLinear)]