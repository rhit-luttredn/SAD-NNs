#!/usr/bin/env python3
from __future__ import annotations

import itertools
import os
import random
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import sad_nns.envs
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.vector import VectorEnv
from minigrid.core.constants import COLOR_NAMES
from minigrid.wrappers import ImgObsWrapper
from neurops.models import ModConv2d, ModLinear, ModSequential
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
# from stable_baselines3 import PPO, DQN
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.dqn.policies import CnnPolicy
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.utils import set_random_seed
# from torch import nn


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SimpleEnv-v0"
    # env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 150_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Experiment specific arguments
    total_architectures: int = 30
    """the number of architectures to test"""

    # Environment specific arguments
    env_size: int = 10
    """the size of the environment"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    # architectures: list = None
    # """the list of architectures to test (computed in runtime)"""
    # training_log: list = None
    # """the list of training logs (computed in runtime)"""
    # parameter_log: list = None
    # """the list of parameter logs (computed in runtime)"""
    # neuron_log: list = None
    # """the list of neuron logs (computed in runtime)"""
    device: torch.device = None
    """the device to run the experiment on (computed in runtime)"""


architectures = []
training_log = []
parameter_log = []
neuron_log = []



def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", size=args.env_size)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, size=args.env_size)
        env = ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs: VectorEnv, network = None, network_output_size: int = 512):
        super().__init__()
        if network is not None != network_output_size is not None:
            raise ValueError("Please provide both `network` and `network_output_size` or none of them.")

        if network is not None:
            height = envs.single_observation_space.shape[0]
            width = envs.single_observation_space.shape[1]
            n_input_channels = envs.single_observation_space.shape[2]

            conv_layers = nn.Sequential(
                layer_init(nn.Conv2d(n_input_channels, 16, 2, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, 2, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 2, padding=1)),
                nn.ReLU(),
            )

            # Calculate the size of the output of the conv_layers by doing one forward pass
            dummy_input = torch.randn(1, n_input_channels, height, width)
            output = conv_layers(dummy_input)
            output_size = output.shape[1] * output.shape[2] * output.shape[3]

            self.network = nn.Sequential(
                conv_layers,
                nn.Flatten(),
                nn.Linear(output_size, 512),
                nn.ReLU(),
            )

            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)
        else:
            self.network = network
            self.actor = layer_init(nn.Linear(network_output_size, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(network_output_size, 1), std=1)


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


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, arch_idx, features_dim: int = 512, normalized_image: bool = False) -> None:
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
        intermediate_conv_sizes = [(8, 16, 32), (16, 32, 64), (32, 64, 128)]  # Each tuple is one possible set of sizes
        intermediate_linear_sizes = [64, 128, 256]  # Each tuple is one possible set of sizes
        models = generate_fixed_structure_models(n_input_channels, 64, intermediate_conv_sizes, intermediate_linear_sizes, 2000)
        model = models[arch_idx]
        architectures.append(model)
        total_params = count_parameters(model)
        parameter_log.append(total_params)
        self.cnn = model

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class PrintAverageRewardCallback(BaseCallback):
    def __init__(self, print_freq, arch_idx):
        super(PrintAverageRewardCallback, self).__init__(verbose=0)
        self.print_freq = print_freq
        self.arch_idx = arch_idx
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
                training_log[self.arch_idx].append(average_reward)
                # print(f"Episode {self.episode_count}: Average Reward: {average_reward}")

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


def generate_fixed_structure_models(input_channels, output_features, intermediate_conv_sizes, intermediate_linear_sizes, max_intermediate_neurons, max_total_params=None):
    models = []

    # Generate all possible combinations of intermediate layers
    for conv_sizes in itertools.product(*intermediate_conv_sizes):
        for linear_size in intermediate_linear_sizes:
            # Check if the total number of neurons does not exceed the maximum
            total_neurons = sum(conv_sizes) + linear_size 

            # Skip if too many neurons
            if total_neurons > max_intermediate_neurons:
                continue

            layers = []

            # Add convolutional layers
            in_channels = input_channels
            for out_channels in conv_sizes:
                layers.append(layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2))))
                layers.append(nn.ReLU())
                in_channels = out_channels

            # Flatten the output of the last conv layer
            layers.append(nn.Flatten())

            # Add the two linear layers
            in_features = in_channels * 16
            layers.append(layer_init(nn.Linear(in_features, linear_size)))

            # Add the final output layer
            layers.append(nn.Linear(linear_size, output_features))

            model = nn.Sequential(*layers)
            
            total_params = count_parameters(model)

            if max_total_params is None or total_params <= max_total_params:
                neuron_log.append(total_neurons)
                parameter_log.append(total_params)
                models.append(model)

    return models


def train_single_model(agent: Agent, optimizer, envs: VectorEnv, writer: SummaryWriter, args: Args):
    device = args.device

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()


def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    n_input_channels = envs.single_observation_space.shape[2]
    output_features = 64
    intermediate_conv_sizes = [(8, 16, 32), (16, 32, 64), (32, 64, 128)]
    intermediate_linear_sizes = [64, 128, 256]
    models = generate_fixed_structure_models(n_input_channels, output_features, intermediate_conv_sizes, intermediate_linear_sizes, 2000)

    start_time = time.time()
    for i, model in enumerate(models):
        print(f'Iteration {i}/{args.total_architectures}')
        
        writer = SummaryWriter(f"../runs/{run_name}/model-{i}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        training_log.append([])
        agent = Agent(envs, model=model, network_output_size=output_features).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        train_single_model(agent, optimizer, envs, writer, args)


        # policy_kwargs = dict(
        #     features_extractor_class=MinigridFeaturesExtractor,
        #     features_extractor_kwargs=dict(arch_idx=i, features_dim=128),
        # )
        # callback = PrintAverageRewardCallback(print_freq=100, arch_idx=i)
        # model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, )
        # model.learn(2e4, callback=callback)


    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # # model.save('DQN-1')

    # test_env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array") 
    # test_env = ImgObsWrapper(test_env)

    # test_model(model, test_env)
        
    # for i in range(total_architectures):
    #     print(architectures[i])
    #     print(parameter_log[i])
    #     print(training_log[i])
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    x1 = parameter_log
    y = [np.mean(sub_array) for sub_array in training_log]

    # Creating a scatter plot
    ax1.scatter(x1, y)
    ax1.set_xlabel('# Total Parameters')
    ax1.set_ylabel('Average Training Reward')
    ax1.set_title('Parameter Impact')
    ax1.grid(True)

    x2 = neuron_log[:args.total_architectures]

    # Creating a scatter plot
    ax2.scatter(x2, y)
    ax2.set_xlabel('# Total Neurons')
    ax2.set_ylabel('Average Training Reward')
    ax2.set_title('Neuron Impact')
    ax2.grid(True)

    plt.show()

        
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time to run file: {} seconds".format(total_time))


if __name__ == "__main__":
    main()