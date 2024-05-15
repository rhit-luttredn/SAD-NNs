# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import sad_nns.envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
# from torch.nn import LSTM
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


@dataclass
class EnvArgs:
    # action_low=[0,-.001,0.0,.4,0.0,0.0]
    # action_high=[1,.1,0.6,1.0,1.0,1.0]
    action_low=[0,-.1,0]
    action_high=[1,.1,1]
    # To be computed at runtime
    device = None

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
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "RegressionEnv-v0"
    """the id of the environment"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: torch.Tensor = field(default_factory=lambda: torch.tensor([0.02,0.0002,0.02],device=device)) #IDK just trying to make it match
    """the scale of policy noise"""
    exploration_noise: torch.Tensor = field(default_factory=lambda: torch.tensor([0.01,0.0001,0.01],device=device))
    """the scale of exploration noise"""
    learning_starts: int = 5000 #Matching again #250#25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: torch.Tensor = field(default_factory=lambda: torch.tensor([0.05,0.0005,0.05],device=device))
    """noise clip parameter of the Target Policy Smoothing Regularization"""


def make_env(env_id, seed, idx, capture_video, run_name, env_kwargs: dict = {}):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):#, lstm_hidden_size=256):
        super().__init__()
        
        # CHAT GPT
        # self.lstm = LSTM(input_size=np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
        #                  hidden_size=lstm_hidden_size,
        #                  batch_first=True)
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, a):
        # CHAT GPT
        x = torch.cat([x, a], 1)
        # x, _ = self.lstm(x.unsqueeze(1))  # Unsqueeze to add a batch dimension
        # x = F.relu(x[:, -1, :])  # Take the last time step's output
        x = self.fc1(x)
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):#, lstm_hidden_size=128):
        super().__init__()
        
        # #CHAT GPT
        # self.lstm = LSTM(input_size=np.array(env.single_observation_space.shape).prod(),
        #                  hidden_size=lstm_hidden_size,
        #                  batch_first=True)
        # self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(),256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        #CHAT GPT
        # x, _ = self.lstm(x.unsqueeze(1))  # Unsqueeze to add a batch dimension
        # x = F.relu(x[:, -1, :])  # Take the last time step's output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    env_args = vars(EnvArgs())
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device = torch.device("cuda:2" if torch.cuda.is_available() and args.cuda else "cpu")
    env_args['device'] = device
    # args.exploration_noise.to(device)
    # args.policy_noise.to(device)
    # args.noise_clip.to(device)
    
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name, env_args)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        noise_factor = global_step/args.total_timesteps
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                # #CHAT GPT
                # # Unsqueeze to add a time dimension for LSTM input
                # obs_with_time = torch.Tensor(obs).unsqueeze(0).unsqueeze(1).to(device)
                # actions = actor(obs_with_time)
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise * noise_factor)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

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
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                
                # #CHAT GPT
                
                # # Unsqueeze to add a time dimension for LSTM input
                # data_observations = data.observations.unsqueeze(1)
                # data_next_observations = data.next_observations.unsqueeze(1)
                
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale * noise_factor

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # print(next_q_value)

            # #CHAT GPT
            # # Unsqueeze to add a time dimension for LSTM input
            # data_observations = data.observations.unsqueeze(1)
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            if np.isinf(qf1_loss.cpu().detach().numpy()).any():
                qf1_loss = F.l1_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            if np.isinf(qf2_loss.cpu().detach().numpy()).any():
                qf2_loss = F.l1_loss(qf1_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                # #CHAT GPT
                # # Unsqueeze to add a time dimension for LSTM input
                # data_observations = data.observations.unsqueeze(1)
                
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")
        from sad_nns.utils.cleanrl.evals.td3_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=0,#args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from sad_nns.utils.cleanrl.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "TD3", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()