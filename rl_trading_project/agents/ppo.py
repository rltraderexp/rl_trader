"""Production-polished PPO agent with multi-dimensional action support.

Enhancements (compared to original lightweight PPO):
- Optional policy types: 'mlp' (default), 'conv1d' (fast local temporal), 'transformer' (global attention).
- Optional compile_policy flag to call torch.compile() when available.
- Optional mixed-precision (use_amp) support for inference/act.
- Back-compat: if you supply flattened observations, for transformer/conv set seq_len & feat_dim so obs can be reshaped.
- Minimal positional encoding included for transformer.

Original implementation used as baseline and preserved behavior by default.
"""
import os, math, random
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import Agent


# -----------------------
# original MLP policy
# -----------------------
class PolicyNetMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(64,64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last,h)); layers.append(nn.ReLU()); last = h
        self.body = nn.Sequential(*layers)
        self.mean = nn.Linear(last, action_dim)
        self.logstd = nn.Parameter(torch.zeros(action_dim))
    def forward(self, x):
        x = self.body(x)
        mean = self.mean(x)
        std = torch.exp(self.logstd)
        return mean, std

# -----------------------
# Small Transformer encoder head and positional encoding
# -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd dimension: last column remains zeros for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TransformerPolicyNet(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, action_dim: int = 1):
        super().__init__()
        # map features to d_model
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max(512, seq_len))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool across seq dim after projection
        self.mean_head = nn.Linear(d_model, action_dim)
        self.logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor):
        """
        x: tensor shaped (B, seq_len, feat_dim)
        returns mean (B, action_dim), std (action_dim)
        """
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.transformer(x)  # (B, S, d_model)
        # pool along sequence
        x_pooled = x.mean(dim=1)  # (B, d_model)
        mean = self.mean_head(x_pooled)
        std = torch.exp(self.logstd)
        return mean, std

# -----------------------
# Conv1D policy for efficient local temporal modeling
# -----------------------
class ConvPolicyNet(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, hidden_channels: int = 64, kernel=3, action_dim: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_channels, kernel_size=kernel, padding=kernel//2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel, padding=kernel//2),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mean = nn.Linear(hidden_channels, action_dim)
        self.logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        # x: (B, seq_len, feat_dim) -> permute to (B, feat_dim, seq_len)
        x = x.permute(0, 2, 1)
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        mean = self.mean(h)
        std = torch.exp(self.logstd)
        return mean, std

# -----------------------
# Value net for sequence inputs (works with all policy types)
# -----------------------
class ValueNet(nn.Module):
    def __init__(self, input_dim: int = None, seq_len: int = None, feat_dim: int = None, hidden=(64,64), mode='mlp'):
        super().__init__()
        self.mode = mode
        if mode == 'mlp':
            layers = []
            last = int(input_dim)
            for h in hidden:
                layers.append(nn.Linear(last, h)); layers.append(nn.ReLU()); last = h
            layers.append(nn.Linear(last, 1))
            self.net = nn.Sequential(*layers)
        else:
            # for sequence modes: use a small conv encoder + linear head
            c_in = feat_dim
            hidden_ch = hidden[0] if hidden else 64
            self.encoder = nn.Sequential(
                nn.Conv1d(c_in, hidden_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.head = nn.Linear(hidden_ch, 1)

    def forward(self, x):
        if self.mode == 'mlp':
            return self.net(x)
        else:
            # x is (B, seq_len, feat_dim)
            h = x.permute(0,2,1)
            h = self.encoder(h).squeeze(-1)
            return self.head(h)
    
# -----------------------
# PPO Agent
# -----------------------
class PPOAgent(Agent):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 clip_eps: float = 0.2,
                 lam: float = 0.95,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.0,
                 epochs: int = 10,
                 minibatch_size: int = 64,
                 max_grad_norm: float = 0.5,
                 lr_scheduler: Optional[dict] = None,
                 seed: Optional[int] = None,
                 device: str = 'cpu',
                 policy_type: str = 'mlp',
                 seq_len: Optional[int] = None,
                 feat_dim: Optional[int] = None,
                 compile_policy: bool = False,
                 use_amp: bool = False):
        """
        New args:
            policy_type: 'mlp' (default), 'conv1d', or 'transformer'
            seq_len, feat_dim: required for 'conv1d' or 'transformer' so flattened obs can be reshaped.
            compile_policy: attempt torch.compile(policy) if available (PyTorch 2.0+)
            use_amp: use mixed precision for inference (autocast) where beneficial
        """
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.policy_type = policy_type
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.use_amp = bool(use_amp)

        # build policy net depending on requested type
        if policy_type == 'mlp':
            self.policy = PolicyNetMLP(self.obs_dim, self.action_dim).to(device)
            self.value = ValueNet(input_dim=self.obs_dim, mode='mlp').to(device)
        elif policy_type == 'conv1d':
            assert seq_len is not None and feat_dim is not None, "seq_len & feat_dim required for conv1d policy"
            self.policy = ConvPolicyNet(seq_len=seq_len, feat_dim=feat_dim, action_dim=self.action_dim).to(device)
            self.value = ValueNet(seq_len=seq_len, feat_dim=feat_dim, mode='seq', hidden=(64,)).to(device)
        elif policy_type == 'transformer':
            assert seq_len is not None and feat_dim is not None, "seq_len & feat_dim required for transformer policy"
            self.policy = TransformerPolicyNet(seq_len=seq_len, feat_dim=feat_dim, d_model=64, nhead=4, num_layers=2, action_dim=self.action_dim).to(device)
            self.value = ValueNet(seq_len=seq_len, feat_dim=feat_dim, mode='seq', hidden=(64,)).to(device)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        # optional compile
        if compile_policy and hasattr(torch, 'compile'):
            try:
                self.policy = torch.compile(self.policy)
            except Exception:
                pass

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

        # usual hyperparams
        self.gamma = float(gamma); self.clip_eps = float(clip_eps); self.lam = float(lam)
        self.value_coef = float(value_coef); self.entropy_coef = float(entropy_coef)
        self.epochs = int(epochs); self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else 0.0

        self.scheduler = None
        if lr_scheduler is not None and isinstance(lr_scheduler, dict):
            if lr_scheduler.get('type') == 'step':
                step_size = int(lr_scheduler.get('step_size', 100))
                gamma = float(lr_scheduler.get('gamma', 0.9))
                self.scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=step_size, gamma=gamma)

    # helper: reshape flat obs to sequence if needed
    def _prepare_obs_for_policy(self, obs_np: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if self.policy_type == 'mlp':
            return t.view(1, -1)
        else:
            # accept either already sequence-shaped or flattened vector that can be reshaped
            if t.ndim == 1:
                assert self.seq_len is not None and self.feat_dim is not None, "seq_len & feat_dim required to reshape flattened obs"
                t = t.view(1, self.seq_len, self.feat_dim)
            elif t.ndim == 2 and t.shape[0] == self.seq_len and t.shape[1] == self.feat_dim:
                t = t.unsqueeze(0)
            elif t.ndim == 2 and t.shape[0] == 1 and t.shape[1] == (self.seq_len * self.feat_dim):
                t = t.view(1, self.seq_len, self.feat_dim)
            elif t.ndim == 3:
                pass
            else:
                # last resort: flatten to 1D and reshape
                t = t.view(1, -1)
                t = t.view(1, self.seq_len, self.feat_dim)
            return t

    def act(self, obs, deterministic: bool = False):
        """
        Returns tuple (action, logp, value).
        Backwards-compatible: accepts flattened obs (1D) or already structured (seq_len, feat_dim).
        """
        obs_t = self._prepare_obs_for_policy(obs)
        with torch.no_grad():
            if self.use_amp:
                # inference in autocast to use mixed precision if available and beneficial
                try:
                    from torch.cuda.amp import autocast
                    ctx = autocast if self.device.startswith('cuda') else nullcontext
                except Exception:
                    # fallback
                    from contextlib import nullcontext
                    ctx = nullcontext
            else:
                from contextlib import nullcontext
                ctx = nullcontext

            with ctx():
                mean, std = self.policy(obs_t if self.policy_type == 'mlp' else obs_t)
                dist = torch.distributions.Normal(mean, std)
                raw_action = mean if deterministic else dist.rsample()
                logp_t = dist.log_prob(raw_action).sum(axis=-1)
                logp = float(logp_t.item())
                value = float(self.value(obs_t if self.policy_type == 'mlp' else obs_t).item())

        action = raw_action.cpu().numpy().flatten()
        action = np.clip(action, -1.0, 1.0)
        return action, logp, value

    # compute GAE (kept from original)
    def compute_gae(self, rewards, values, dones):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            next_value = values[t+1] if t+1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values[:len(adv)]
        return adv, returns

    def update(self, trajectories):
        # Mostly unchanged from original: we reshape obs into tensors and run PPO epochs.
        if len(trajectories) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        obs = np.vstack([t['obs'] for t in trajectories]).astype(np.float32)
        acts = np.array([t['act'] for t in trajectories], dtype=np.float32).reshape(-1, self.action_dim)
        rews = np.array([t['rew'] for t in trajectories], dtype=np.float32)
        dones = np.array([t['done'] for t in trajectories], dtype=np.float32)
        old_logps = np.array([t.get('logp', 0.0) for t in trajectories], dtype=np.float32)

        # compute values (vectorized)
        with torch.no_grad():
            if self.policy_type == 'mlp':
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                values = self.value(obs_t).cpu().numpy().flatten()
            else:
                # reshape obs into (N, seq_len, feat_dim)
                N = obs.shape[0]
                obs_seq = obs.reshape(N, self.seq_len, self.feat_dim)
                obs_t = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
                values = self.value(obs_t).cpu().numpy().flatten()

        values_ext = np.concatenate([values, np.array([0.0], dtype=np.float32)], axis=0)
        advs, returns = self.compute_gae(rews, values_ext, dones)
        adv_mean, adv_std = advs.mean(), advs.std()
        advs = (advs - adv_mean) / (adv_std + 1e-8)

        # Prepare tensors for optimization
        if self.policy_type == 'mlp':
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            N = obs.shape[0]
            obs_t = torch.as_tensor(obs.reshape(N, self.seq_len, self.feat_dim), dtype=torch.float32, device=self.device)
        acts_t = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns.reshape(-1,1), dtype=torch.float32, device=self.device)
        advs_t = torch.as_tensor(advs.reshape(-1,1), dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_logps.reshape(-1,1), dtype=torch.float32, device=self.device)

        N = len(trajectories)
        batch_size = min(self.minibatch_size, N)
        idxs = np.arange(N)

        policy_loss_total, value_loss_total, num_updates = 0.0, 0.0, 0
        policy_grad_norm_total, value_grad_norm_total = 0.0, 0.0
        for epoch in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, batch_size):
                mb_idx = idxs[start:start+batch_size]
                mb_obs, mb_acts = obs_t[mb_idx], acts_t[mb_idx]
                mb_returns, mb_advs = returns_t[mb_idx], advs_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]

                means, stds = self.policy(mb_obs if self.policy_type == 'mlp' else mb_obs)
                dist = torch.distributions.Normal(means, stds)
                logp = dist.log_prob(mb_acts).sum(dim=-1, keepdim=True)

                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist.entropy().mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                # Calculate total norm of gradients
                policy_grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in self.policy.parameters() if p.grad is not None))
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_pred = self.value(mb_obs if self.policy_type == 'mlp' else mb_obs)
                value_loss = nn.functional.mse_loss(value_pred, mb_returns) * self.value_coef
                value_loss.backward()
                # Calculate total norm of gradients
                value_grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in self.value.parameters() if p.grad is not None))
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                policy_grad_norm_total += float(policy_grad_norm.item())
                value_grad_norm_total += float(value_grad_norm.item())
                num_updates += 1

        if self.scheduler is not None:
            self.scheduler.step()

        avg_policy_loss = policy_loss_total / max(1, num_updates)
        avg_value_loss = value_loss_total / max(1, num_updates)
        avg_policy_grad = policy_grad_norm_total / max(1, num_updates)
        avg_value_grad = value_grad_norm_total / max(1, num_updates)
        return {'policy_loss': avg_policy_loss, 'value_loss': avg_value_loss, 
                'policy_grad_norm': avg_policy_grad, 'value_grad_norm': avg_value_grad}
    
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, 'policy.pth'))
        torch.save(self.value.state_dict(), os.path.join(path, 'value.pth'))

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))
        self.value.load_state_dict(torch.load(os.path.join(path, 'value.pth')))