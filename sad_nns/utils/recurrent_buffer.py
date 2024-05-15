import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.buffers import BaseBuffer

class ReccurentBuffer(BaseBuffer):
    '''
    Replay buffer for off policy learning with recurrent agents
    Based on the stable baselines replay buffer
    
    '''
    
    # head: int
    tail: int
    
 
    observations: np.ndarray
    # next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    # timeouts: np.ndarray
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_run_length: int,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        # optimize_memory_usage: bool = False, # Memory will be optimized by default as I already need some extra complexity to retrieve the rollout
        # handle_timeout_termination: bool = True, # I cannot be bothered to deal with timeout, we are not using it.
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        if max_run_length>self.buffer_size:
            raise Exception("Max run length greater than buffer length. Runs may not be able to be stored.")
        
        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        # if optimize_memory_usage and handle_timeout_termination:
        #     raise ValueError(
        #         "ReplayBuffer does not support optimize_memory_usage = True "
        #         "and handle_timeout_termination = True simultaneously."
        #     )
        # self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        # if not optimize_memory_usage:
        #     # When optimizing memory, `observations` contains also the next observation
        # self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        # self.handle_timeout_termination = handle_timeout_termination
        # self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # self.head = 0
        self.tail = 0
        
        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            # if not optimize_memory_usage:
                # total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        self.pos += 1
        self.pos %= self.buffer_size
        
        
        # move the tail to the end of the next run to make room for data
        if self.pos == self.tail:
            while True:
                self.tail+= 1
                self.tail%= self.buffer_size
                if self.dones[self.tail].all():
                    break
                if self.tail == self.pos:
                    raise Exception("Tail reached head while trying to find end of run. This means that the run was longer than the buffer. Decrease max run length or increase buffer size to solve this issue.")
        self.observations[self.pos] = np.array(next_obs)
        
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # if not self.optimize_memory_usage:
        #     return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        # if self.full:
        #     batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        # else:
        #     batch_inds = np.random.randint(0, self.pos, size=batch_size)
        # return self._get_samples(batch_inds, env=env)
        # batch_inds = (np.random.randint())
        records = self.pos-self.tail
        if(records<0):
            records += self.buffer_size
        #TODO    
        