import numpy as np
import torch

from minigrid.wrappers import FullyObsWrapper

class FullyObsRotatingWrapper(FullyObsWrapper):
    def observation(self, obs):
        obs = super().observation(obs)
        # obs['image'] = torch.from_numpy(obs['image']).permute(2, 0, 1).numpy()
        obs['image'] = np.rot90(obs['image'], -obs['direction'], axes=(0, 1))
        # obs['image'] = torch.from_numpy(obs['image']).permute(1, 2, 0).numpy()
        return obs
