from __future__ import annotations

import itertools

import gymnasium as gym
import numpy as np
import sad_nns.envs
import torch
from minigrid.core.constants import COLOR_NAMES
from minigrid.wrappers import ImgObsWrapper
from neurops.models import ModConv2d, ModLinear, ModSequential
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.dqn.policies import CnnPolicy
from torch import nn


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 16, (2, 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, (2, 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (2, 2)),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        # models = generate_models(n_input_channels, 64, 25000, 5000)
        models = generate_neuron_balanced_models()
        model = models[2]
        print(model)
        total_params = count_parameters(model)
        print(total_params)
        self.cnn = model

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
class PrintAverageRewardCallback(BaseCallback):
    def __init__(self, print_freq):
        super(PrintAverageRewardCallback, self).__init__(verbose=0)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_count = 0
        self.cumulative_reward = 0

    def _on_step(self) -> bool:
        # Retrieve reward and done flag
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        # Update cumulative reward
        self.cumulative_reward += reward

        # Check if episode is done
        if done:
            self.episode_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0
            self.episode_count += 1

            # Print average reward
            if self.episode_count % self.print_freq == 0:
                average_reward = sum(self.episode_rewards[-self.print_freq:]) / self.print_freq
                print(f"Episode {self.episode_count}: Average Reward: {average_reward}")

        return True

def convert_matrix_format(matrix):
    """
    Converts a 3D matrix of shape (7, 7, 3) to a 4D matrix of shape (1, 3, 7, 7).

    :param matrix: A 3D NumPy array of shape (7, 7, 3).
    :return: A 4D NumPy array of shape (1, 3, 7, 7).
    """
    # Check if the input matrix has the expected shape
    if matrix.shape != (7, 7, 3):
        raise ValueError("Input matrix must have shape (7, 7, 3)")

    # Transpose the matrix from shape (7, 7, 3) to (3, 7, 7)
    transposed_matrix = np.transpose(matrix, (2, 0, 1))

    # Add an extra dimension to get shape (1, 3, 7, 7)
    reshaped_matrix = np.expand_dims(transposed_matrix, axis=0)

    return reshaped_matrix

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model(model, env, num_episodes=50):
    for episode in range(num_episodes):
        obs = env.reset()[0]
        done = False
        truncated = False
        total_reward = 0

        while (not truncated and not done):
            # obs_image = obs['image']
            obs = convert_matrix_format(obs)
            # print(obs)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            

            # env.render()  # Optional: Render the environment to visualize

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# generate based on total parameters
def generate_models(n_input_channels, n_output_channels, total_params, tolerance=500):
    models = []
    
    # Helper function to calculate parameters for a Conv2d layer
    def conv2d_params(in_channels, out_channels, kernel_size):
        return (kernel_size[0] * kernel_size[1] * in_channels + 1) * out_channels

    # Define possible layer sizes for intermediate layers
    intermediate_layer_sizes = [16, 32, 64, 128]

    # Generate different model configurations
    for size1 in intermediate_layer_sizes:
        for size2 in intermediate_layer_sizes:
            # Calculate the total parameters for this configuration
            params = conv2d_params(n_input_channels, size1, (2, 2)) \
                   + conv2d_params(size1, size2, (2, 2)) \
                   + conv2d_params(size2, n_output_channels, (2, 2))

            # Check if the parameters are within the tolerance
            if abs(params - total_params) <= tolerance:
                model = nn.Sequential(
                    nn.Conv2d(n_input_channels, size1, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(size1, size2, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(size2, n_output_channels, (2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                models.append(model)

    return models
        
def generate_neuron_balanced_models(input_size, output_size, total_neurons, layer_options):
    models = []

    # Helper function to calculate total neurons for a given configuration
    def total_neurons_in_config(layers):
        total = input_size + sum(layers) + output_size
        return total

    # Generate all possible combinations of layers
    for L in range(1, len(layer_options) + 1):
        for subset in itertools.combinations_with_replacement(layer_options, L):
            if total_neurons_in_config(subset) == total_neurons:
                layers = [nn.Linear(input_size, subset[0])]
                layers += [nn.Linear(subset[i], subset[i+1]) for i in range(len(subset) - 1)]
                layers += [nn.Linear(subset[-1], output_size)]
                model = nn.Sequential(*layers)
                models.append(model)

    return models


def main():

    set_random_seed(0)
    
    # manual control of environment
    # env = ToughEnv(size=10, render_mode="human")
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array") 
    env = ImgObsWrapper(env)

    callback = PrintAverageRewardCallback(print_freq=100)

    # model = DQN.load('DQN-1', env=env)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, )
    model.learn(2e4, callback=callback)

    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # # model.save('DQN-1')

    # test_env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array") 
    # test_env = ImgObsWrapper(test_env)

    # test_model(model, test_env)


if __name__ == "__main__":
    main()