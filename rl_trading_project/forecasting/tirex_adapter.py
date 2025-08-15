"""TiRexAdapter: adapter for using TiRex forecasts inside the RL pipeline.

Behavior:
- If the `tirex` module is importable and `model_path` is provided, this adapter will attempt to
  load and use TiRex for predictions. Because TiRex usage and checkpoints can vary, the loading
  code is left as a clear placeholder for you to adapt to your local TiRex setup.
- If TiRex is not available or loading fails, the adapter falls back to a deterministic stub
  which returns a simple drifted forecast based on the last close price.
- Predictions are cached in-memory to avoid repeated inference during training loops.
"""

from typing import Dict, Any, Optional, Tuple
import importlib, hashlib, json
import numpy as np
import pandas as pd
from collections import OrderedDict

def _history_key(history: pd.DataFrame, horizon: int) -> str:
    # Create a lightweight cache key from the last few closes and horizon
    closes = history['close'].astype(float).values[-16:].tolist()  # last up to 16 values
    payload = {'closes': closes, 'horizon': int(horizon)}
    s = json.dumps(payload, separators=(',', ':'), sort_keys=True)
    return hashlib.sha1(s.encode('utf8')).hexdigest()

class TiRexAdapter:
    def __init__(self, model_path: Optional[str]=None, device: str='cpu', cache_size: int=128):
        """
        model_path: path to a TiRex model checkpoint (optional)
        device: 'cpu' or 'cuda'
        cache_size: in-memory cache entries
        """
        self.model_path = model_path
        self.device = device
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()
        self._model = None
        self._tirex = None
        self._available = False
        # try import
        try:
            self._tirex = importlib.import_module('tirex')
            self._available = True
        except ImportError:
            self._tirex = None
            self._available = False

        # if model_path provided and tirex importable, attempt to load model (user must customize loader)
        if self._available and self.model_path is not None:
            try:
                self._load_model(self.model_path)
            except Exception as e:
                # fail gracefully and leave model as None (stub will be used)
                print(f"TiRexAdapter: failed to load model at {self.model_path}: {e}")
                self._model = None

    def _load_model(self, model_path: str):
        """
        Placeholder: customize this method to load your TiRex model from checkpoint.
        The exact API depends on the TiRex version and model serialization used.
        Example (pseudo-code):
            from tirex import ModelClass
            self._model = ModelClass.load_from_checkpoint(model_path, map_location=self.device)
            self._model.to(self.device).eval()
        """
        # This is intentionally a placeholder to be adapted locally.
        raise NotImplementedError("Please implement _load_model() to load TiRex checkpoints for your environment.")

    def _cache_get(self, key: str):
        if key in self._cache:
            # move to end (LRU)
            val = self._cache.pop(key)
            self._cache[key] = val
            return val
        return None

    def _cache_set(self, key: str, val: Dict[str, Any]):
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False) # Evict oldest
        self._cache[key] = val

    def predict(self, history: pd.DataFrame, horizon: int = 60) -> Dict[str, Any]:
        """
        history: DataFrame with at least a 'close' column and most recent rows last.
        horizon: integer forecast horizon (number of steps forward; units should match your env e.g. minutes)
        Returns dict with keys:
            - 'mean': list of floats, length == horizon
            - 'quantiles': dict mapping quantile string to list of floats
        """
        key = _history_key(history, horizon)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # If a real TiRex model is available and loaded, run it.
        if self._model is not None:
            try:
                # Placeholder call: replace with your model's inference API.
                # Example expected return: numpy array shape (horizon,) or (num_samples, horizon)
                preds = self._model.predict(history, horizon=horizon, device=self.device)
                
                if np.ndim(preds) == 2: # Probabilistic forecast
                    means = np.asarray(preds).mean(axis=0).tolist()
                    quantiles = {
                        '0.1': np.quantile(preds, 0.1, axis=0).tolist(),
                        '0.5': np.quantile(preds, 0.5, axis=0).tolist(),
                        '0.9': np.quantile(preds, 0.9, axis=0).tolist()
                    }
                else: # Point forecast
                    means = np.asarray(preds).tolist()
                    quantiles = {'0.5': means} # Use mean as median
                
                out = {'mean': means, 'quantiles': quantiles}
                self._cache_set(key, out)
                return out
            except Exception as e:
                print(f"TiRexAdapter: model inference failed, falling back to stub: {e}")

        # Fallback deterministic stub: linearized small random walk anchored at last close
        last = float(history['close'].iloc[-1])
        rng = np.random.RandomState(int(sum(map(int, np.asarray(history['close'].astype(float)[-4:])))) & 0xffffffff)
        steps = np.arange(1, horizon + 1)
        drift = 0.0001  # small drift per step
        noise_vol = 0.001 * last
        noise = rng.normal(scale=noise_vol, size=horizon).cumsum()
        means = (last * (1.0 + drift * steps) + noise).tolist()
        
        std_dev = np.sqrt(steps) * noise_vol
        quantiles = {
            '0.1': (np.array(means) - 1.282 * std_dev).tolist(),
            '0.5': means,
            '0.9': (np.array(means) + 1.282 * std_dev).tolist()
        }
        out = {'mean': means, 'quantiles': quantiles}
        self._cache_set(key, out)
        return out

    def batch_predict(self, histories: List[pd.DataFrame], horizon: int = 60) -> List[Dict[str, Any]]:
        """
        Batch predict multiple history windows in one call. Returns list of prediction dicts matching histories order.
        Behavior:
          - If a model is loaded and supports batched inference, use it.
          - Otherwise fall back to per-history predict() (still cached) or the stub.
        Notes:
          - This reduces python-to-model call overhead and is recommended when you gather many windows (e.g., vectorized envs).
        """
        results = []
        # Create keys and check cache first
        keys = [_history_key(h, horizon) for h in histories]
        cached = [self._cache_get(k) for k in keys]

        # If all cached -> return
        if all(cached):
            return list(cached)

        # If underlying model supports batched inference, attempt to call it
        if self._model is not None:
            try:
                # expected: model.predict accepts list/array of windows -> returns array shape (batch, horizon) or (nsamples, batch, horizon)
                # This is adapter-dependent; modify to your model's API.
                raw_preds = self._model.predict_many(histories, horizon=horizon, device=self.device)  # PSEUDO-call: customize locally
                # Normalize outputs into list-of-dicts
                if isinstance(raw_preds, np.ndarray):
                    if raw_preds.ndim == 2:
                        for i in range(raw_preds.shape[0]):
                            means = raw_preds[i].tolist()
                            out = {'mean': means, 'quantiles': {'0.5': means}}
                            self._cache_set(keys[i], out)
                            results.append(out)
                    elif raw_preds.ndim == 3:
                        # shape: (nsamples, batch, horizon) -> compute quantiles
                        ns, batch, hor = raw_preds.shape
                        for j in range(batch):
                            mat = raw_preds[:, j, :]
                            means = mat.mean(axis=0).tolist()
                            quantiles = {
                                '0.1': np.quantile(mat, 0.1, axis=0).tolist(),
                                '0.5': np.quantile(mat, 0.5, axis=0).tolist(),
                                '0.9': np.quantile(mat, 0.9, axis=0).tolist()
                            }
                            out = {'mean': means, 'quantiles': quantiles}
                            self._cache_set(keys[j], out)
                            results.append(out)
                    else:
                        # fallback: per-window
                        for h, k in zip(histories, keys):
                            out = self.predict(h, horizon)
                            results.append(out)
                else:
                    # unknown type -> fallback to per-window predict
                    for h in histories:
                        results.append(self.predict(h, horizon))
            except Exception:
                # if model doesn't provide batch API, fallback to per-window
                for h in histories:
                    results.append(self.predict(h, horizon))
        else:
            # no model -> fallback to per-window predict (which uses stub with caching)
            for h in histories:
                results.append(self.predict(h, horizon))
        return results