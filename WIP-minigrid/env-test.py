from dataclasses import dataclass

import sad_nns.envs
import tyro
from gymnasium import make


@dataclass
class Args:
    env_id: str = "HardWall-v0"
    """the environment id"""
    render: str = "human"
    """the rendering mode"""
    env_size: int|None = 10
    """the height and width of the environment"""
    env_width: int|None = None
    """the width of the environment"""
    env_height: int|None = None
    """the height of the environment"""
    agent_start_pos: tuple|None = None,
    """the starting position of the agent"""
    agent_start_dir: int = 0,
    """the starting direction of the agent"""
    env_max_steps: int|None = None,
    """the maximum number of steps before the environment is terminated"""
    see_through_walls: bool = True,
    """whether the agent can see through walls"""
    wall_freq: int = 2,
    """FOR HARDWALL: the number of tiles between walls"""
    use_lava: bool = False,
    """FOR HARDWALL: whether to use lava"""
    lock_doors: bool = False,
    """FOR HARDWALL: whether to lock doors"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args = vars(args)
    env_id = args.pop("env_id")
    render = args.pop("render")
    env = make(env_id, render=render, **args)
