#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from itertools import product

import gymnasium as gym
import numpy as np
import pandas as pd
import sad_nns.envs
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.vector import VectorEnv
from minigrid.core.constants import COLOR_NAMES
from minigrid.wrappers import ImgObsWrapper
from neurops.models import ModConv2d, ModLinear, ModSequential
from sad_nns.modules import PPOAgent
from sad_nns.utils.cleanrl.evals.ppo_eval import evaluate
from torch.distributions.categorical import Categorical
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


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
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "WallEnv-v0"
    """the id of the environment"""
    total_timesteps: int = 50_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 256
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
    output_features: int = 64
    """the number of output features for each model's feature extractor"""
    # conv_sizes: list = field(default_factory=lambda: [(8, 16), (16, 32)])
    conv_sizes: list = field(default_factory=lambda: [(8, 16, 32), (16, 32, 64), (32, 64, 128)])
    """the possible output channels for the each convolutional layers. Each tuple is a set for one layer"""
    # linear_sizes: list = field(default_factory=lambda: [[64], [64], [64]])
    # linear_sizes: list = field(default_factory=lambda: [64])
    linear_sizes: list = field(default_factory=lambda: [(8, 16, 32), (16, 32, 64), (32, 64, 128)])
    """the possible output sizes for the each linear layers. Each tuple is a set for one layer"""
    training_loops: int = 20
    """the number of independent training loops for each experiment"""
    max_architectures: int = 5
    """the total number of architectures to test"""

    # Environment specific arguments
    env_size: int = 10
    """the size of the environment"""
    wall_density: float = 0.5
    use_lava: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    device: torch.device = None
    """the device to run the experiment on (computed in runtime)"""
    env_kwargs: dict = None
    """the additional kwargs to pass to the gym environment (computed in runtime)"""


parameter_log = []
neuron_log = []


def make_env(env_id, idx, capture_video, run_name, env_kwargs: dict = {}):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env = gym.wrappers.RecordVideo(env, f"../videos/{run_name}")
        else:
            env = gym.make(env_id, **env_kwargs)
        env = ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# class Agent(nn.Module):
#     def __init__(self, envs: VectorEnv, network = None, network_output_size: int = 512):
#         super().__init__()
#         if (network is not None) != (network_output_size is not None):
#             raise ValueError("Please provide both `network` and `network_output_size` or none of them.")

#         if network is None:
#             height = envs.single_observation_space.shape[0]
#             width = envs.single_observation_space.shape[1]
#             n_input_channels = envs.single_observation_space.shape[2]

#             conv_layers = nn.Sequential(
#                 layer_init(nn.Conv2d(n_input_channels, 16, 2, padding=1)),
#                 nn.ReLU(),
#                 layer_init(nn.Conv2d(16, 32, 2, padding=1)),
#                 nn.ReLU(),
#                 layer_init(nn.Conv2d(32, 64, 2, padding=1)),
#                 nn.ReLU(),
#             )

#             # Calculate the size of the output of the conv_layers by doing one forward pass
#             dummy_input = torch.randn(1, n_input_channels, height, width)
#             output = conv_layers(dummy_input)
#             output_size = output.shape[1] * output.shape[2] * output.shape[3]

#             self.network = nn.Sequential(
#                 conv_layers,
#                 nn.Flatten(),
#                 nn.Linear(output_size, 512),
#                 nn.ReLU(),
#             )

#             self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
#             self.critic = layer_init(nn.Linear(512, 1), std=1)
#         else:
#             self.network = network
#             self.actor = layer_init(nn.Linear(network_output_size, envs.single_action_space.n), std=0.01)
#             self.critic = layer_init(nn.Linear(network_output_size, 1), std=1)

#     def get_value(self, x):
#         x = x.permute(0, 3, 1, 2)
#         return self.critic(self.network(x / 255.0))

#     def get_action_and_value(self, x, action=None):
#         x = x.permute(0, 3, 1, 2)
#         hidden = self.network(x / 255.0)
#         logits = self.actor(hidden)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def generate_fixed_structure_models(input_channels, output_features, possible_conv_sizes, possible_linear_sizes):
#     models = []
#     # TODO: Implement using multiple linear layers
#     # Generate all possible combinations of intermediate layers
#     for conv_sizes in product(*possible_conv_sizes):
#         for linear_size in possible_linear_sizes:
#             # Check if the total number of neurons does not exceed the maximum
#             total_neurons = sum(conv_sizes) + linear_size 

#             layers = []

#             # Add convolutional layers
#             in_channels = input_channels
#             for out_channels in conv_sizes:
#                 layers.append(layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2))))
#                 layers.append(nn.ReLU())
#                 in_channels = out_channels

#             # Flatten the output of the last conv layer
#             layers.append(nn.Flatten())

#             # Add the two linear layers
#             in_features = in_channels * 16
#             layers.append(layer_init(nn.Linear(in_features, linear_size)))

#             # Add the final output layer
#             layers.append(nn.Linear(linear_size, output_features))

#             model = nn.Sequential(*layers)
            
#             total_params = count_parameters(model)

#             neuron_log.append(total_neurons)
#             parameter_log.append(total_params)
#             models.append(model)

#     return models


def train_single_model(agent, optimizer, envs: VectorEnv, writer: SummaryWriter, args: Args):
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


def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.env_kwargs = {
        "size": args.env_size,
        "use_lava": args.use_lava,
    }
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

    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = args.device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, env_kwargs=args.env_kwargs) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # n_input_channels = envs.single_observation_space.shape[2]
    # architectures = generate_fixed_structure_models(
    #     n_input_channels,
    #     args.output_features,
    #     args.conv_sizes,
    #     args.linear_sizes,
    # )

    df = pd.DataFrame(
        columns=[
            "# Total Parameters", 
            "# Total Neurons", 
            "Model Summary", 
            "Environment", 
            "Size",
            "Use Lava",
            "Average Test Reward", 
            "Test Reward Variance"
        ])

    # Generate all possible combinations of architectures
    # arch_params = [args.conv_sizes, args.linear_sizes]
    # architectures = list(product(*[product(*x) for x in arch_params]))
    architectures = list(product(*args.linear_sizes))
    if args.max_architectures:
        architectures = random.sample(architectures, args.max_architectures)

    # Iterate through each architecture
    start_time = time.time()
    model_rewards = []
    for i, linear_sizes in enumerate(architectures):
    # for i, model in enumerate(architectures):
        print(f'Iteration {i}/{len(architectures) - 1}')
        model_kwargs = {
            "conv_sizes": [8, 16, 32],
            "linear_sizes": linear_sizes,
            "out_features": args.output_features,
        }
        # model_kwargs = {
        #     "network": model,
        #     "network_output_size": args.output_features,
        # }
        
        # print(model_kwargs)
        agent = PPOAgent(envs, **model_kwargs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        print(agent.network)
        parameter_log.append(count_parameters(agent.network))
        neuron_log.append(sum(p.numel() for p in agent.network.parameters() if p.requires_grad))

        for train_loop in range(args.training_loops):
            print(f"Iteration: {train_loop}")
            print("Training...")
            writer = SummaryWriter(f"../runs3/{run_name}/model-{i}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )

            train_single_model(agent, optimizer, envs, writer, args)

            # Test the model
            print("Testing...")
            model_path = f"../runs3/{run_name}/model-{i}/{args.exp_name}.model"
            torch.save(agent.state_dict(), model_path)
            print(f"model {i} saved to {model_path}")

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=PPOAgent,
                device=device,
                capture_video=True,
                env_kwargs=args.env_kwargs,
                model_kwargs=model_kwargs,
            )
            model_rewards.append(np.mean(episodic_returns))
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)
            
            writer.close()

        df.loc[len(df.index)] = {
            "# Total Parameters": parameter_log[i],
            "# Total Neurons": neuron_log[i],
            "Model Summary": str(agent.network),
            "Environment": args.env_id,
            "Size": args.env_size,
            "Use Lava": args.use_lava,
            "Average Test Reward": np.mean(model_rewards),
            "Test Reward Variance": np.std(model_rewards)
        }

    envs.close()

    df.to_csv(f"../runs3/{run_name}/results.csv", index=False)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # x1 = parameter_log
    # y = [np.mean(sub_array) for sub_array in model_rewards]

    # # Creating a scatter plot
    # ax1.scatter(x1, y)
    # ax1.set_xlabel('# Total Parameters')
    # ax1.set_ylabel('Average Training Reward')
    # ax1.set_title('Parameter Impact')
    # ax1.grid(True)

    # x2 = neuron_log

    # # Creating a scatter plot
    # ax2.scatter(x2, y)
    # ax2.set_xlabel('# Total Neurons')
    # ax2.set_ylabel('Average Training Reward')
    # ax2.set_title('Neuron Impact')
    # ax2.grid(True)

    # plt.show()

    # end_time = time.time()
    # total_time = end_time - start_time
    # print("Total time to run file: {} seconds".format(total_time))


if __name__ == "__main__":
    main()