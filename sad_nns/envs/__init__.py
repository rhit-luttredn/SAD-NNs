from sad_nns.envs.simple import SimpleEnv
from sad_nns.envs.wall import WallEnv
from gymnasium import register

register(
    id='SimpleEnv-v0',
    entry_point='sad_nns.envs:SimpleEnv',
)

register(
    id='WallEnv-v0',
    entry_point='sad_nns.envs:WallEnv',
)