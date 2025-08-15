"""Simple replay buffer for off-policy algorithms"""
import numpy as np
from typing import Tuple

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int]):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.ptr = 0
        self.size = 0
        self.obs_buf = np.zeros((capacity,)+self.obs_shape, dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity,)+self.obs_shape, dtype=np.float32)
        self.acts_buf = np.zeros((capacity,), dtype=np.float32)
        self.rews_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.obs_buf[idxs],
                self.acts_buf[idxs],
                self.rews_buf[idxs],
                self.next_obs_buf[idxs],
                self.done_buf[idxs])

    def __len__(self):
        return self.size