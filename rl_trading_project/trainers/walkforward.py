"""Walk-forward backtesting harness with optional retraining per fold."""
from typing import Callable, Dict, Any, List, Optional
import numpy as np
import os
import pandas as pd
from .backtester import Backtester, save_history_csv, compare_strategies

class WalkForwardBacktester:
    def __init__(self, env_factory_fn: Callable[[int, int], Callable[[], Any]],
                 train_size: int = 200, test_size: int = 50, step: int = 50):
        """
        env_factory_fn(start_idx, end_idx) -> callable env factory for that segment.
        """
        self.env_factory_fn = env_factory_fn
        self.train_size = int(train_size)
        self.test_size = int(test_size)
        self.step = int(step)

    def _get_folds(self, start_index: int, end_index: int, max_folds: Optional[int]=None):
        folds = []
        current_start = start_index
        while current_start + self.train_size + self.test_size <= end_index:
            if max_folds is not None and len(folds) >= max_folds:
                break
            train_end = current_start + self.train_size
            test_end = train_end + self.test_size
            folds.append({
                'train_start': current_start, 'train_end': train_end,
                'test_start': train_end, 'test_end': test_end
            })
            current_start += self.step
        return folds

    def run(self, policy_map: Dict[str, Callable], start_index: int, end_index: int, max_folds: Optional[int]=None, save_dir: Optional[str]=None):
        """Run standard walk-forward backtest for given policies."""
        folds = self._get_folds(start_index, end_index, max_folds)
        all_fold_results = []

        for i, fold in enumerate(folds):
            print(f"Running Fold {i+1}/{len(folds)}: Test Period {fold['test_start']}-{fold['test_end']}")
            fold_results = {}
            for name, policy in policy_map.items():
                env_factory = self.env_factory_fn(fold['test_start'], fold['test_end'])
                bt = Backtester(env_factory, start_index=fold['test_start'])
                res = bt.run(policy, max_steps=self.test_size)
                fold_results[name] = res
                
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    path = os.path.join(save_dir, f"fold_{i}_{name}_history.csv")
                    save_history_csv(res.get('history', []), path)
            
            all_fold_results.append(compare_strategies(fold_results))

        # Aggregate and average metrics across all folds
        mean_metrics = {}
        if all_fold_results:
            agg_df = pd.DataFrame()
            for i, fold_res in enumerate(all_fold_results):
                df = pd.DataFrame(fold_res).T
                df['fold'] = i
                agg_df = pd.concat([agg_df, df])
            
            for policy_name in policy_map.keys():
                policy_df = agg_df.loc[agg_df.index == policy_name]
                mean_metrics[policy_name] = policy_df.mean(numeric_only=True).to_dict()

        return {'folds': all_fold_results, 'mean_metrics': mean_metrics}

    def run_with_retraining(self, policy_map: Dict[str, Callable], start_index: int, end_index: int, 
                            retrain_fn: Callable, agent_policy_name: str, 
                            max_folds: Optional[int]=None, save_dir: Optional[str]=None):
        """Run walk-forward with policy retraining on each fold."""
        folds = self._get_folds(start_index, end_index, max_folds)
        # ... Implementation for retraining would be similar to run(), but would call
        # a `retrain_fn` before evaluating on the test set. `retrain_fn` would
        # take the training data segment and return a new trained policy function.
        # This is left as an extension.
        print("run_with_retraining is not fully implemented in this version.")
        return {}