import numpy as np

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper, ObservationWrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from gymnasium import spaces


class FullyObsRotatingWrapper(FullyObsWrapper):
    def observation(self, obs):
        obs = super().observation(obs)
        # obs['image'] = torch.from_numpy(obs['image']).permute(2, 0, 1).numpy()
        obs['image'] = np.rot90(obs['image'], -obs['direction'], axes=(0, 1))
        # obs['image'] = torch.from_numpy(obs['image']).permute(1, 2, 0).numpy()
        return obs


class OneHotImageWrapper(ObservationWrapper):
    def __init__(self, env: MiniGridEnv):
        super().__init__(env)

        self.observation_space = env.observation_space

        orig_image_shape = env.observation_space.spaces['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(orig_image_shape[0], orig_image_shape[1], len(OBJECT_TO_IDX) + len(COLOR_TO_IDX)),
            dtype="uint8",
        )
    
    def observation(self, obs):
        img_obs = obs['image']
        object_one_hot = np.eye(len(OBJECT_TO_IDX), dtype='uint8')
        color_one_hot = np.eye(len(COLOR_TO_IDX), dtype='uint8')

        # Apply the one-hot encoding to the object and color indices
        object_encoded = object_one_hot[img_obs[..., 0]]
        color_encoded = color_one_hot[img_obs[..., 1]]

        # Concatenate along the last dimension to combine object and color encodings
        img_obs = np.concatenate((object_encoded, color_encoded), axis=-1)
        obs['image'] = img_obs

        return obs
