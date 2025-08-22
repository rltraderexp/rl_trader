"""Dueling DQN implementation (PyTorch) where network outputs Q-values for all discrete action bins."""
import os, math, random
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from .base import Agent
from .prioritized_replay_buffer import PrioritizedReplayBuffer

class DuelingDQNNet(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 action_bins: int = 11, 
                 hidden_sizes: Tuple[int, ...] = (128,128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.body = nn.Sequential(*layers)
        # value head
        self.value_head = nn.Sequential(nn.Linear(last, 64), nn.ReLU(), nn.Linear(64, 1))
        # advantage head outputs action_bins advantages
        self.adv_head = nn.Sequential(nn.Linear(last, 64), nn.ReLU(), nn.Linear(64, action_bins))

    def forward(self, x):
        x = self.body(x)
        v = self.value_head(x)            # (B,1)
        a = self.adv_head(x)             # (B, action_bins)
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)             # (B, action_bins)
        return q

class DuelingDQNAgent(Agent):
    def __init__(self, 
                 obs_dim: int, 
                 action_bins: int = 11, 
                 hidden=(128,128), 
                 lr=1e-3, 
                 gamma=0.99, 
                 buffer_size=50000, 
                 batch_size=64,
                 per_alpha=0.6, 
                 per_beta_start=0.4, 
                 per_beta_anneal_steps=100000, 
                 seed: Optional[int] = None, 
                 device='cpu'):
        """
        Action is discretized into `action_bins` in [-1,1]. Network outputs a Q-vector of size action_bins.
        Now uses Prioritized Experience Replay.
        """
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.obs_dim = int(obs_dim)
        self.action_bins = int(action_bins)
        self.device = device
        self.q_net = DuelingDQNNet(self.obs_dim, action_bins=self.action_bins, hidden_sizes=hidden).to(self.device)
        self.target_q = DuelingDQNNet(self.obs_dim, action_bins=self.action_bins, hidden_sizes=hidden).to(self.device)
        self.target_q.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_count = 0
        self.replay = PrioritizedReplayBuffer(buffer_size, (self.obs_dim,), alpha=per_alpha, beta=per_beta_start, beta_anneal_steps=per_beta_anneal_steps)
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 1e-4
        # discrete actions mapping
        self.action_values = np.linspace(-1.0, 1.0, self.action_bins).astype(np.float32)

    def _obs_to_tensor(self, obs: np.ndarray):
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(-1, self.obs_dim)

    def act(self, obs, deterministic: bool = False):
        # epsilon-greedy over discrete action bins
        if (not deterministic) and (np.random.rand() < self.eps):
            idx = np.random.randint(0, self.action_bins)
            return float(self.action_values[idx])
        with torch.no_grad():
            t = self._obs_to_tensor(obs)
            q = self.q_net(t)  # (1, action_bins)
            idx = int(torch.argmax(q, dim=1).cpu().numpy()[0])
            return float(self.action_values[idx])

    def add_experience(self, obs, act, rew, next_obs, done):
        self.replay.add(np.asarray(obs, dtype=np.float32).reshape(self.obs_dim,), float(act), float(rew), np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim,), bool(done))

    def update(self, sync_freq: int = 100):
        if len(self.replay) < max(self.batch_size, 1000):
            return {'loss': None}
        obs_b, act_b, rew_b, next_obs_b, done_b, is_weights, tree_idxs = self.replay.sample(self.batch_size)

        obs_b = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device)           # (B, obs_dim)
        act_b = torch.as_tensor(act_b, dtype=torch.float32, device=self.device)           # (B,)
        rew_b = torch.as_tensor(rew_b, dtype=torch.float32, device=self.device).unsqueeze(-1)   # (B,1)
        next_obs_b = torch.as_tensor(next_obs_b, dtype=torch.float32, device=self.device) # (B, obs_dim)
        done_b = torch.as_tensor(done_b, dtype=torch.float32, device=self.device).unsqueeze(-1) # (B,1)
        is_weights = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(-1) # (B,1)

        # Compute Q(s,a) by indexing the q_net outputs at the action bin indices
        q_all = self.q_net(obs_b)
        action_values = torch.as_tensor(self.action_values, dtype=torch.float32, device=self.device)
        dist = torch.abs(act_b.unsqueeze(1) - action_values.unsqueeze(0))
        idx = torch.argmin(dist, dim=1)
        q_sa = q_all.gather(1, idx.unsqueeze(1))

        # target: reward + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next_all = self.target_q(next_obs_b)
            q_next_max, _ = q_next_all.max(dim=1, keepdim=True)
            q_target = rew_b + (1 - done_b) * self.gamma * q_next_max
            td_errors = (q_target - q_sa).squeeze().cpu().numpy()

        # Update priorities in replay buffer
        self.replay.update_priorities(tree_idxs, td_errors)
        
        # Compute loss with importance sampling weights
        loss = (is_weights * nn.functional.mse_loss(q_sa, q_target, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % sync_freq == 0:
            self.target_q.load_state_dict(self.q_net.state_dict())

        # decay epsilon
        self.eps = max(self.eps_min, self.eps - self.eps_decay)
        return {'loss': float(loss.item()), 'eps': float(self.eps), 'beta': self.replay.beta}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join(path, 'q_net.pth'))
        torch.save(self.target_q.state_dict(), os.path.join(path, 'target_q.pth'))

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(os.path.join(path, 'q_net.pth')))
        self.target_q.load_state_dict(torch.load(os.path.join(path, 'target_q.pth')))