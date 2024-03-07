import random

import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Lava, Door, Key
from minigrid.minigrid_env import MiniGridEnv


class MineFieldEnv(MiniGridEnv):
    def __init__(
        self,
        size: int|None = None,
        width: int|None = None,
        height: int|None = None,
        agent_start_pos: tuple|None = None,
        agent_start_dir: int = 0,
        max_steps: int|None = None,
        see_through_walls: bool = True,
        wall_density: int = 0.5,
        use_lava: bool = True,
        **kwargs,
    ):
        if size is not None:
            assert width is None and height is None
            width = size
            height = size
        assert width is not None and height is not None

        if max_steps is None:
            max_steps = 8 * height * width

        assert wall_density < 1

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.wall_density = wall_density
        self.use_lava = use_lava

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=see_through_walls,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
        else:
            self.agent_pos = (1, int(height/2))
        self.agent_dir = self.agent_start_dir

        # Place a goal square in the bottom-right corner
        self.goal_pos = (width-2, height-2)
        self.put_obj(Goal(), *self.goal_pos)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate random grid of 1s and 0s in shape of grid
        grid = np.random.choice([0, 1], size=(height-2, width-2), p=[1-self.wall_density, self.wall_density])

        # Make a random path to prevent an impossible env
        path = self.generate_random_path(
            width - 2, 
            height - 2, 
            (self.agent_pos[0] - 1, self.agent_pos[1] - 1),
            (self.goal_pos[0] - 1, self.goal_pos[1] - 1)
        )
        grid = grid * path

        # Generate walls where there is a one
        wall_obj = Wall() if not self.use_lava else Lava()
        rows, cols = np.where(grid==1)
        positions = np.column_stack((rows, cols))
        for pos in positions:
            self.put_obj(wall_obj, pos[1]+1, pos[0]+1)

        self.mission = "grand mission"

    def generate_random_path(self, width, height, start, end):
        # Initialize the grid with ones
        grid = np.ones((height, width), dtype=int)

        current = start
        while current != end:
            # Mark the current position with zero
            grid[current[1], current[0]] = 0

            # Determine potential next steps
            next_steps = [tuple(map(sum, zip(current, direction))) for direction in [(1,0), (0,1), (0,-1)]]
            valid_next_steps = [step for step in next_steps if 0 <= step[0] < width and 0 <= step[1] < height]
            # valid_next_steps = [step for step in valid_next_steps if grid[step] != 0]

            # Randomly choose the next step
            if len(valid_next_steps) > 1:
                current = random.choice(valid_next_steps)
            else:
                current = valid_next_steps[0]

        # Mark the end position with zero
        grid[end[1], end[0]] = 0

        return grid
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            assert reward > 0, "The agent should have reached the goal."
            reward = reward / 2 + 0.5
        elif truncated:
            assert reward == 0, "The agent should NOT have reached the goal."
            # calc manhattan distance to goal
            dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            reward = 1 / (dist + 1)
            reward /= 2

        return obs, reward, terminated, truncated, info