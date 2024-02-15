from gymnasium import register
from sad_nns.envs.hardwall import HardWallEnv
from sad_nns.envs.minefield import MineFieldEnv
from sad_nns.envs.simple import SimpleEnv
from sad_nns.envs.wall import WallEnv
from sad_nns.envs.hporegression import RegressionEnv

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
    id='RegressionEnv-v0',
    entry_point='sad_nns.envs:RegressionEnv',
)