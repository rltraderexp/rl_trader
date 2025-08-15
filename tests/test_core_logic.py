"""
Unit tests for critical project components.
This suite verifies:
- The logic of the PrioritizedReplayBuffer.
- The correctness of the Black-Scholes pricing functions.
- The margin call mechanism in the PortfolioEnv.
- The basic functioning of the various PPO agent policy heads.
"""
import unittest
import numpy as np
import pandas as pd
from rl_trading_project.agents.prioritized_replay_buffer import PrioritizedReplayBuffer
from rl_trading_project.options.black_scholes import bs_price, bs_delta
from rl_trading_project.envs.portfolio_env import PortfolioEnv
from rl_trading_project.agents.ppo import PPOAgent

class TestCoreLogic(unittest.TestCase):

    def test_prioritized_replay_buffer(self):
        buffer = PrioritizedReplayBuffer(capacity=128, obs_shape=(4,), alpha=0.6, beta=0.4)
        for i in range(150):
            buffer.add(np.ones(4)*i, i, i, np.ones(4)*i, False)

        self.assertEqual(len(buffer), 128)
        
        obs, act, rew, next_obs, done, weights, indices = buffer.sample(32)
        self.assertEqual(obs.shape, (32, 4))
        self.assertEqual(weights.shape, (32,))
        self.assertEqual(indices.shape, (32,))

        td_errors = np.random.rand(32)
        buffer.update_priorities(indices, td_errors)
        self.assertIsNotNone(buffer.tree.total())

    def test_black_scholes_values(self):
        # Known values from an online calculator for:
        # S=100, K=100, r=0.05, sigma=0.2, tau=1.0 (1 year)
        price_call = bs_price(S=100, K=100, r=0.05, sigma=0.2, tau=1.0, option_type='call')
        delta_call = bs_delta(S=100, K=100, r=0.05, sigma=0.2, tau=1.0, option_type='call')
        
        # Expected values (approximate)
        self.assertAlmostEqual(price_call, 10.45, delta=0.01)
        self.assertAlmostEqual(delta_call, 0.6368, delta=0.001)

    def test_portfolio_env_margin_call(self):
        # Create a DataFrame where a trade at t=1 is followed by a price crash at t=2
        timestamps = pd.to_datetime(['2025-01-01 10:00', '2025-01-01 10:01', '2025-01-01 10:02', '2025-01-01 10:03'])
        data = {
            'timestamp': list(timestamps) * 2,
            'asset': ['A']*4 + ['B']*4,
            'close': [100, 100, 50, 50] + [200, 200, 200, 200] # Asset A buys at 100, then price drops to 50
        }
        df = pd.DataFrame(data).set_index(['timestamp', 'asset'])
        df['open'] = df['high'] = df['low'] = df['close']
        df['volume'] = 100

        # High leverage, low maintenance margin threshold to make triggering easier
        env = PortfolioEnv(df, window_size=1, initial_balance=10000, max_leverage=5.0, maintenance_margin_ratio=0.8)
        
        # Start at index 1, where price is 100
        obs, _ = env.reset(start_index=1)
        
        # Go max long on the volatile asset A at price 100
        action = np.array([1.0, 0.0]) # Target 5x leverage on Asset A
        
        # The environment will now step to index 2, where the price is 50, causing a large loss
        obs, reward, terminated, truncated, info = env.step(action)

        # After the price drop, the position should be liquidated
        self.assertTrue(info['margin_called'])
        self.assertTrue(terminated)
        self.assertAlmostEqual(info['positions'][0], 0.0) # Position in asset A should be zeroed
        self.assertAlmostEqual(info['total_value'], info['cash']) # Total value should equal cash after liquidation

    def test_ppo_policy_networks(self):
        obs_dim_flat = 128
        seq_len, feat_dim = 16, 8
        action_dim = 2
        
        # MLP
        agent_mlp = PPOAgent(obs_dim=obs_dim_flat, action_dim=action_dim, policy_type='mlp')
        obs_mlp = np.random.rand(obs_dim_flat)
        act, _, _ = agent_mlp.act(obs_mlp)
        self.assertEqual(act.shape, (action_dim,))

        # Conv1D
        agent_conv = PPOAgent(obs_dim=seq_len*feat_dim, action_dim=action_dim, policy_type='conv1d', seq_len=seq_len, feat_dim=feat_dim)
        obs_conv = np.random.rand(seq_len, feat_dim)
        act, _, _ = agent_conv.act(obs_conv)
        self.assertEqual(act.shape, (action_dim,))

        # Transformer
        agent_tf = PPOAgent(obs_dim=seq_len*feat_dim, action_dim=action_dim, policy_type='transformer', seq_len=seq_len, feat_dim=feat_dim)
        obs_tf = np.random.rand(seq_len, feat_dim)
        act, _, _ = agent_tf.act(obs_tf)
        self.assertEqual(act.shape, (action_dim,))

if __name__ == '__main__':
    unittest.main()