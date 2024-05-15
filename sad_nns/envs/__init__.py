from gymnasium import register
from sad_nns.envs.hardwall import HardWallEnv
from sad_nns.envs.minefield import MineFieldEnv
from sad_nns.envs.simple import SimpleEnv
from sad_nns.envs.wall import WallEnv
from sad_nns.envs.door import DoorEnv
from sad_nns.envs.wall2 import WallEnv2
from sad_nns.envs.hporegression import RegressionEnv, Regression_Env_no_time

register(
    id='SimpleEnv-v0',
    entry_point='sad_nns.envs:SimpleEnv',
)

register(
    id='WallEnv-v0',
    entry_point='sad_nns.envs:WallEnv',
)

register(
    id='HardWallEnv-v0',
    entry_point='sad_nns.envs:HardWallEnv',
)

register(
    id='MineFieldEnv-v0',
    entry_point='sad_nns.envs:MineFieldEnv',
)

register(
    id='DoorEnv-v0',
    entry_point='sad_nns.envs:DoorEnv'
)

register(
    id='WallEnv2-v0',
    entry_point='sad_nns.envs:WallEnv2'
)
register(
    id='RegressionEnv-v0',
    entry_point='sad_nns.envs:RegressionEnv',
)

register(
    id='RegressionEnv-v1',
    entry_point='sad_nns.envs:Regression_Env_no_time'
)