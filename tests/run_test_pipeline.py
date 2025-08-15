import tempfile
import pandas as pd
from pathlib import Path
from rl_trading_project.data.csv_loader import load_csv
from rl_trading_project.data.transforms.canonicalize import canonicalize_ohlcv

def create_sample_csv(path: Path):
    df = pd.DataFrame({
        'time': ['2025-08-01 00:00:00','2025-08-01 00:01:00','2025-08-01 00:02:00'],
        'OPEN': [100.0, 101.0, 100.5],
        'High': [101.0, 101.5, 101.0],
        'lOw': [99.5, 100.8, 100.2],
        'close': [100.8, 100.9, 100.6],
        'vol': [10, 5, 8]
    })
    df.to_csv(path, index=False)

def run_test(tmp_dir: str):
    tmp_path = Path(tmp_dir)
    csv_path = tmp_path / 'sample.csv'
    create_sample_csv(csv_path)
    
    # Test loading with automatic column name finding
    df_loaded = load_csv(str(csv_path))
    print('Loaded DataFrame:')
    print(df_loaded.head())
    assert 'timestamp' in df_loaded.columns
    assert 'volume' in df_loaded.columns
    
    # Test canonicalization
    df_canonical = canonicalize_ohlcv(df_loaded)
    print('\nCanonicalized DataFrame:')
    print(df_canonical.head())
    assert 'vwap' in df_canonical.columns
    assert 'trade_count' in df_canonical.columns
    assert len(df_canonical) == 3
    assert pd.api.types.is_datetime64_any_dtype(df_canonical['timestamp'])
    
    print('\nData pipeline test passed.')

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        run_test(tmpdir)