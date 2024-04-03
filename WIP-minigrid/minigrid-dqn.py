#!/usr/bin/env python
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import sys
import time
from dataclasses import dataclass
from operator import mul

import gymnasium as gym
import numpy as np
import sad_nns.envs
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.vector import VectorEnv
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ActionBonus
from sad_nns.envs.wrappers import FullyObsRotatingWrapper
from stable_baselines3.common.buffers import ReplayBuffer
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


@dataclass
class EnvArgs:
    size: int|None = 10
    """the height and width of the environment"""
    width: int|None = None
    """the width of the environment"""
    height: int|None = None
    """the height of the environment"""
    agent_start_pos: tuple|None = None
    """the starting position of the agent"""
    agent_start_dir: int = 0
    """the starting direction of the agent"""
    max_steps: int|None = None
    """the maximum number of steps before the environment is terminated"""
    see_through_walls: bool = True
    """whether the agent can see through walls"""

    wall_density: int = 0.3
    use_lava: bool = False

    # wall_freq: int = 2
    # """FOR HARDWALL: the number of tiles between walls"""
    # use_lava: bool = False
    # """FOR HARDWALL: whether to use lava"""
    # lock_doors: bool = False
    # """FOR HARDWALL: whether to lock doors"""


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
    device: int|None = 6
    """the GPU device to use"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "uncertainty-estimation-3"
    """the wandb's project name"""
    wandb_entity: str = 'sad-nns'
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    mc_dropout: bool = True
    """whether to use Monte Carlo Dropout for uncertainty estimation"""

    # Algorithm specific arguments
    env_id: str = "MineFieldEnv-v0"
    """the id of the environment"""
    total_timesteps: int = 150_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the replay memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1_200
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    dropout_frequency: int = 2000
    """the frequency of dropout"""


def make_env(env_id, idx, capture_video, run_name, env_kwargs: dict = {}):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env = gym.wrappers.RecordVideo(env, f"../videos/{run_name}")
        else:
            env = gym.make(env_id, **env_kwargs)
        # env = FullyObsRotatingWrapper(env)
        env = ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, envs: VectorEnv):
        super().__init__()
        height = envs.single_observation_space.shape[0]
        width = envs.single_observation_space.shape[1]
        n_input_channels = envs.single_observation_space.shape[2]
        self.dropout_layers = []

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

        self.network = nn.Sequential(
            conv_layers,
            nn.Flatten(),
            nn.Linear(output_size, 512),
            nn.ReLU(),
            self._make_dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            self._make_dropout(0.5),
            nn.Linear(512, envs.single_action_space.n),
        )

    def _make_dropout(self, p: float):
        dropout = nn.Dropout(p)
        self.dropout_layers.append(dropout)
        return dropout

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.network(x / 255.0)

    def enable_dropout(self):
        for layer in self.dropout_layers:
            layer.train()

    def disable_dropout(self):
        for layer in self.dropout_layers:
            layer.eval()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def mc_dropout(make_envs_thunk: callable, agent: QNetwork, forward_passes: int, eval_episodes: int, device: torch.device, verbose=False):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    envs : VectorEnv
        vectorized environment
    model : Agent
        model to be used for prediction
    forward_passes : int
        number of monte-carlo samples/forward passes
    eval_episodes : int
        number of episodes to evaluate the model
    device : torch.device
        device to be used for computation
    """
    print("Starting Monte Carlo")
    start_time = time.time()
    envs = make_envs_thunk(False)
    num_envs = envs.num_envs
    single_obs_shape = envs.single_observation_space.shape
    # n_classes = envs.single_action_space.n
    softmax = nn.Softmax(dim=-1)
    softmax_temp = 0.1

    # agent.eval()
    obs, _ = envs.reset()
    variances = []
    entropies = []
    mutual_infos = []
    episode_count = 0
    agent.enable_dropout()

    with torch.no_grad():
        while episode_count < eval_episodes:
            # Broadcast the obs to batch our forward passes
            obs = torch.from_numpy(obs).to(device)
            obs = obs.unsqueeze(0).repeat(forward_passes, 1, *([1] * (obs.dim() - 1)))  # shape (forward_passes, num_envs, *single_obs_shape)
            obs = obs.view(forward_passes * num_envs, *single_obs_shape)  # shape (forward_passes * num_envs, *single_obs_shape)

            # MC Dropout
            dropout_vals = agent(obs)  # shape (forward_passes * num_envs, n_classes)
            dropout_vals = dropout_vals.view(forward_passes, num_envs, -1)  # shape (forward_passes, num_envs, n_classes)
            # if verbose:
            #     print("dropout:", dropout_vals)
                # print("mean:", mean)
            dropout_vals = softmax(dropout_vals / softmax_temp)  # shape (forward_passes, num_envs, n_classes)
            # if verbose:
            #     print("softmax:", dropout_vals)
            # assert torch.isfinite(dropout_vals).all(), "dropout_vals has non-finite values"

            # Calculate variance and entropy
            variance = torch.var(dropout_vals, dim=0)  # shape (num_envs, n_classes)
            variances.append(variance)

            mean = torch.mean(dropout_vals, dim=0)  # shape (num_envs, n_classes)
            # if verbose:
            #     print("mean:", mean)
            entropy = -torch.sum(mean * torch.log(mean + 1e-9), axis=-1)  # shape (num_envs,)

            # if verbose:
            #     new = -torch.sum(dropout_vals * torch.log(dropout_vals + 1e-9), axis=-1)
            #     print("new:", new)
            #     print("mean new:", torch.mean(new, axis=0))

            # entropy = -torch.sum(dropout_vals * torch.log(dropout_vals + 1e-9), axis=-1)  # shape (forward_passes, num_envs)
            # if verbose:
            #     print("entropy:", entropy)
            # entropy = torch.mean(entropy, axis=0)  # shape (num_envs,)

            # assert torch.isfinite(entropy).all(), "entropy has non-finite values"
            entropies.append(entropy)

            # mutual_info = entropy - np.mean(np.sum(-dropout_vals * np.log(dropout_vals + epsilon), axis=-1), axis=0)  # shape (num_envs,)
            mutual_info = entropy - torch.mean(torch.sum(-dropout_vals * torch.log(dropout_vals + 1e-8), axis=-1), axis=0)  # shape (num_envs,)
            mutual_infos.append(mutual_info)

            # Take a step
            actions = torch.argmax(dropout_vals, dim=-1)  # shape (forward_passes, num_envs)
            actions = actions.permute(1, 0)  # shape (num_envs, forward_passes)
            actions = torch.mode(actions, dim=1).values.cpu().numpy()
            next_obs, _, _, _, infos = envs.step(actions)

            if "final_info" in infos:
                episode_count += sum("episode" in info for info in infos["final_info"])

            obs = next_obs

        # Aggregate the results
        variances = torch.cat(variances, dim=0).to(device)  # shape (eval_episodes * num_envs, n_classes)
        variance = torch.mean(variances, dim=0)  # shape (n_classes,)
        entropies = torch.cat(entropies, dim=0).to(device)  # shape (eval_episodes * num_envs,)
        entropy = torch.mean(entropies)  # shape (1,)
        mutual_infos = torch.cat(mutual_infos, dim=0).to(device)  # shape (eval_episodes * num_envs,)
        mutual_info = torch.mean(mutual_infos)  # shape (1,)

    print("Monte Carlo took", time.time() - start_time, "seconds")
    return entropy.item(), variance, mutual_info.item()


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    env_args = vars(EnvArgs())
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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
    writer = SummaryWriter(f"../runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda", args.device if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    def make_envs_thunk(capture_video=args.capture_video):
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, capture_video, run_name, env_kwargs=env_args) for i in range(args.num_envs)]
        )
        return envs
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name, env_kwargs=env_args) for i in range(args.num_envs)]
    # )
    envs = make_envs_thunk()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # Uncertainty Estimation
        if global_step % args.dropout_frequency == 0:
            entropy, variance, mutual_info = mc_dropout(make_envs_thunk, q_network, forward_passes=10, eval_episodes=args.num_envs, device=device)

            explore_dist = torch.distributions.Categorical(probs=variance)
            print(f"entropy={entropy}, variance={variance}, mutual_info={mutual_info}")
            print(f"epsilon={entropy / np.log(envs.single_action_space.n)}, explore_dist={explore_dist.probs}")

            for i in range(len(variance)):
                writer.add_scalar(f"uncertainty/variance_{i}", variance[i], global_step)
            writer.add_scalar("uncertainty/avg_variance", variance.mean(), global_step)
            writer.add_scalar("uncertainty/entropy", entropy, global_step)

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        # epsilon = entropy / np.log(envs.single_action_space.n)
        if random.random() < epsilon:
            if not args.mc_dropout:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions = explore_dist.sample(sample_shape=(envs.num_envs,)).cpu().numpy()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length {info['episode']['l']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"../runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from sad_nns.utils.cleanrl.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            env_kwargs=env_args
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()