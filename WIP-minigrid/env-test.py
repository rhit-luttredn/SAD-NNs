#! /usr/bin/env python3
from dataclasses import dataclass

from typing import Tuple, Union
import sad_nns.envs
import tyro
from gymnasium import make
from minigrid.manual_control import ManualControl

@dataclass
class Args:
    env_id: str = "HardWallEnv-v0"
    # env_id: str = "MineFieldEnv-v0"
    """the environment id"""
    render_mode: str = "human"
    """the rendering mode"""
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
    
    wall_density: int = 0.5
    use_lava: bool = False

    # wall_freq: int = 2
    # """FOR HARDWALL: the number of tiles between walls"""
    # use_lava: bool = False
    # """FOR HARDWALL: whether to use lava"""
    # lock_doors: bool = False
    # """FOR HARDWALL: whether to lock doors"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args = vars(args)

    env_id = args.pop("env_id")
    args = {k: v for k, v in args.items() if v is not None}
    env = make(env_id, **args)
    if args["render_mode"] == "human":
        manual_control = ManualControl(env)
        manual_control.start()

