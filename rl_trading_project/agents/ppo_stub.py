"""PPO stub wrapper - placeholder for future Stable-Baselines3 integration."""
from .base import Agent

class PPOAgent(Agent):
    def __init__(self):
        raise NotImplementedError("PPO wrapper not implemented in lightweight demo. Install stable-baselines3 or implement custom PPO.")

    def act(self, obs, deterministic=False):
        pass
    def update(self, *args, **kwargs):
        pass
    def save(self, path):
        pass
    def load(self, path):
        pass