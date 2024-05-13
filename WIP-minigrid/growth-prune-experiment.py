#!/usr/bin/env python3
import json
import pathlib
import os
import random
import subprocess
import sys
import tempfile
import time
from copy import copy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import sad_nns.envs
import torch
import tyro
from sad_nns.modules import QNetwork
from sad_nns.utils import make_env


@dataclass
class FrozenBabyArgs:
    num_runs: int = 3
    """the number of runs for the frozen baby experiment"""
    total_timesteps: int = 150_000
    """the total number of timesteps for each run (unless early stopping)"""
    early_stop: bool = False
    """whether to stop early if the agent solves the environment"""
    growth: bool = False
    """if toggled, the network will grow"""
    prune: bool = False
    """if toggled, the network will prune"""
    tags: tuple[str, ...] = ('frozen-baby', 'static', 'baby')
    """the tags of this experiment"""


@dataclass
class GrowingBabyArgs:
    num_runs: int = 3
    """the number of runs for the growing baby experiment"""
    total_timesteps: int = 150_000
    """the total number of timesteps to train **ONCE FROZEN** each run (unless early stopping)."""
    early_stop: bool = False
    """whether to stop early if the agent solves the environment"""
    growth: bool = True
    """if toggled, the network will grow"""
    prune: bool = True
    """if toggled, the network will prune"""
    tags: tuple[str, ...] = ('growing-baby', 'growth', 'baby', 'adult')
    """the tags of this experiment"""

    # To be determined at runtime
    init_weights: pathlib.Path|None = None
    """the path to the initial weights"""
    stop_growth: int|None = None
    """if not None, the network will stop growing at this step"""


@dataclass
class FrozenAdultArgs:
    num_runs: int = 3
    """the number of runs for the frozen adult experiment (per architecture)"""
    total_timesteps: int = 150_000
    """the total number of timesteps for each run (unless early stopping)"""
    early_stop: bool = False
    """whether to stop early if the agent solves the environment"""
    growth: bool = False
    """if toggled, the network will grow"""
    prune: bool = False
    """if toggled, the network will prune"""
    tags: tuple[str, ...] = ('frozen-adult', 'static', 'adult')
    """the tags of this experiment"""

    # To be determined at runtime
    linear_sizes: tuple[int, ...] = ()
    """the hidden sizes of the fully connected layers"""


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] + "-" + time.strftime("%Y%m%d-%H%M%S")
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    devices: int|tuple = 6
    """the device numbers to use"""
    procs_per_device: int = 1
    """the number of processes to run per device"""

    wandb_project_name: str|None = None
    """the wandb's project name, if None, use exp_name"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    mc_dropout: bool = False
    """whether to use Monte Carlo Dropout for uncertainty estimation"""
    env_id: str = "MineFieldEnv-v0"
    """the id of the environment"""
    linear_sizes: tuple[int, ...] = (256, 256, 128)
    """the hidden sizes of the fully connected layers"""


def make_command(args: dict):
    command = [
        "python3",
        "dqn-north-uncertainty-grow-and-prune.py"
    ]

    # Add all the arguments as command line arguments
    for arg_name, arg_value in args.items():
        arg_name = arg_name.replace("_", "-")

        if isinstance(arg_value, bool):
            arg = f"--{arg_name}" if arg_value else f"--no-{arg_name}"
            command.append(arg)
        elif isinstance(arg_value, tuple):
            command.append(f"--{arg_name}")
            # command.append(arg_name)
            for value in arg_value:
                command.append(str(value))
        # handle none values
        elif arg_value is None:
            pass
        else:
            command.append(f"--{arg_name}")
            command.append(str(arg_value))

    return command


def run_command(cmd: list[str], log_name: str = None):
    print(f"Running: {' '.join(cmd)}")

    if log_name is not None:
        log_path = f"../runs/{args.exp_name}/logs/{log_name}"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            result = subprocess.run(cmd, stdout=f)
    else:
        result = subprocess.run(cmd)

    return result.returncode == 0


VERBOSE = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.wandb_project_name is None:
        args.wandb_project_name = args.exp_name
    
    exp_args = vars(args)
    num_devices = exp_args.pop("devices")
    procs_per_device = exp_args.pop("procs_per_device")

    # Set random seed
    seed = exp_args.pop("seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create a dummy env to create initial weights with
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, False, "", env_kwargs={"size": 10})]
    )

    """Frozen Baby Experiments"""
    frozen_baby_args = copy(exp_args)
    frozen_baby_args.update(vars(FrozenBabyArgs()))

    for run in range(frozen_baby_args.pop('num_runs')):
        with tempfile.NamedTemporaryFile() as f:
            torch.manual_seed(seed + run)
            qnetwork = QNetwork(envs, frozen_baby_args['linear_sizes'])
            torch.save(qnetwork.state_dict(), f)
            frozen_baby_args["init_weights"] = pathlib.Path(f.name)
            frozen_baby_args["seed"] = run

            log_name = None
            if not VERBOSE:
                log_name = f"frozen_baby_{run}.log"
            if not run_command(make_command(frozen_baby_args), log_name=log_name):
                print("Error in running the frozen baby experiment")
                sys.exit(1)

    # Go through the runs and get the global step counts
    step_counts = []
    for dir in pathlib.Path(f"../runs/{args.exp_name}").glob("*"):
        if not dir.is_dir() or dir.name == "logs":
            continue
        run_name = dir.name

        # Check if the run is a frozen baby run
        with open(dir / "args.json") as f:
            run_args = json.load(f)

        if "frozen-baby" not in run_args["tags"]:
            continue

        # Load the run_stats and get the global step count
        with open(dir / "run_stats.json") as f:
            run_stats = json.load(f)
        step_counts.append(run_stats["global_steps"])

    # Determine how long the growing baby has to grow
    grow_time = max(step_counts)
    
    """Growing Baby Experiments"""
    growing_baby_args = copy(exp_args)
    growing_baby_args.update(vars(GrowingBabyArgs()))
    growing_baby_args["total_timesteps"] += grow_time
    growing_baby_args["stop_growth"] = grow_time

    # Save it to a tempfile and pass it to the growing baby runs
    for run in range(growing_baby_args.pop('num_runs')):
        growing_baby_args["seed"] = seed + run
        with tempfile.NamedTemporaryFile() as f:
            torch.manual_seed(seed + run)
            qnetwork = QNetwork(envs, growing_baby_args['linear_sizes'])
            torch.save(qnetwork.state_dict(), f)
            growing_baby_args["init_weights"] = pathlib.Path(f.name)

            log_name = None
            if not VERBOSE:
                log_name = f"growing_baby_{run}.log"
            
            if not run_command(make_command(growing_baby_args), log_name=log_name):
                print("Error in running the growing baby experiment")
                sys.exit(1)

    # Go through the runs and get the final linear sizes
    linear_sizes = []
    for dir in pathlib.Path(f"../runs/{args.exp_name}").glob("*"):
        if not dir.is_dir() or dir.name == "logs":
            continue
        run_name = dir.name

        # Check if the run is a growing baby run
        with open(dir / "args.json") as f:
            run_args = json.load(f)

        if "growing-baby" not in run_args["tags"]:
            continue
        
        # Load the model_kwargs and get the linear sizes
        with open(dir / "model_kwargs.json") as f:
            model_kwargs = json.load(f)
        linear_sizes.append(tuple(model_kwargs["linear_sizes"]))
    
    linear_sizes = set(linear_sizes)

    """Frozen Adult Experiments"""
    for linear_size in linear_sizes:
        frozen_adult_args = copy(exp_args)
        frozen_adult_args.update(vars(FrozenAdultArgs()))
        frozen_adult_args["linear_sizes"] = linear_size

        # Share random set of weights between the runs
        for run in range(frozen_adult_args.pop('num_runs')):
            frozen_adult_args["seed"] = run

            log_name = None
            if not VERBOSE:
                log_name = f"frozen_adult_{'-'.join(str(x) for x in linear_size)}_{run}.log"
            
            if not run_command(make_command(frozen_adult_args), log_name=log_name):
                print("Error in running the frozen adult experiment")
                sys.exit(1)
