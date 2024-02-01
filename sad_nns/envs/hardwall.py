from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Lava, Door, Key
from minigrid.minigrid_env import MiniGridEnv


class HardWallEnv(MiniGridEnv):
    def __init__(
        self,
        size: int|None = None,
        width: int|None = None,
        height: int|None = None,
        agent_start_pos: tuple|None = None,
        agent_start_dir: int = 0,
        max_steps: int|None = None,
        see_through_walls: bool = True,
        wall_freq: int = 2,
        use_lava: bool = False,
        lock_doors: bool = False,
        **kwargs,
    ):
        if size is not None:
            assert width is None and height is None
            width = size
            height = size
        assert width is not None and height is not None

        if max_steps is None:
            max_steps = 8 * height * width

        assert wall_freq > (0 if not lock_doors else 1)

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.wall_freq = wall_freq
        self.use_lava = use_lava
        self.lock_doors = lock_doors

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

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
        else:
            self.agent_pos = (1, int(height/2))
        self.agent_dir = self.agent_start_dir

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Generate verical separation walls
        sep_obj = Wall if not self.use_lava else Lava
        curr_x = self.wall_freq + 1
        while curr_x < width - 2:
            self.grid.vert_wall(curr_x, 1, length=height-2, obj_type=sep_obj)

            door_y = self._rand_int(1, height - 1)
            door_color = self._rand_int(0, len(COLOR_NAMES))
            door = Door(COLOR_NAMES[door_color], is_locked=self.lock_doors)
            self.put_obj(door, curr_x, door_y)

            if self.lock_doors:
                key_loc = self._rand_pos(curr_x - self.wall_freq, curr_x, 1, height - 1)
                while key_loc == self.agent_pos:
                    key_loc = self._rand_pos(curr_x - self.wall_freq, curr_x, 1, height - 1)

                self.put_obj(Key(COLOR_NAMES[door_color]), *key_loc)
            
            curr_x += self.wall_freq + 1

        self.mission = "grand mission"