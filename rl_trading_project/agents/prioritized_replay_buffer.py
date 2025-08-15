"""Prioritized Experience Replay (PER) buffer implementation.
Uses a SumTree data structure for efficient sampling based on TD-error priorities.
Reference: https://arxiv.org/abs/1511.05952
"""
import numpy as np
import random

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: object):
        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple, alpha: float = 0.6, beta: float = 0.4, beta_anneal_steps: int = 100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / beta_anneal_steps
        self.epsilon = 1e-5
        self.max_priority = 1.0

    def add(self, obs, act, rew, next_obs, done):
        experience = (obs, act, rew, next_obs, done)
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size: int):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if data != 0:
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        # Unpack experiences
        obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)

        return (np.array(obs_b), np.array(act_b), np.array(rew_b),
                np.array(next_obs_b), np.array(done_b, dtype=float),
                np.array(is_weights), np.array(idxs))

    def update_priorities(self, tree_indices, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.max_priority = max(self.max_priority, np.max(priorities))
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries