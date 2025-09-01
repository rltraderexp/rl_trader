"""Simple experiment logger that writes metrics to CSV and saves model checkpoints."""
import os
import csv
import json
import time
from pathlib import Path
from typing import Optional # <-- Make sure this line is present

class ExperimentLogger:
    def __init__(self, base_dir: str = "runs", exp_name: Optional[str] = None):
        """
        Initializes the logger.

        Args:
            base_dir (str): The base directory to save all experiment runs.
            exp_name (Optional[str]): A specific name for this experiment. 
                                      If provided, the run folder will be named 
                                      '{exp_name}_{timestamp}'. Otherwise, it will be 'run_{timestamp}'.
        """
        self.base_dir = Path(base_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{exp_name}_{timestamp}" if exp_name else f"run_{timestamp}"
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.csv"
        self._header_written = False
        self._fieldnames = []

    def log_metrics(self, metrics: dict, step: int):
        row = {'step': step, **metrics}
        
        # Ensure all keys are strings and handle non-scalar values
        for k, v in row.items():
            if not isinstance(v, (int, float, str, bool)):
                row[k] = str(v)
        
        # Dynamically update header if new metrics appear
        new_fields = [k for k in row.keys() if k not in self._fieldnames]
        if not self._header_written or new_fields:
            self._fieldnames.extend(new_fields)
            self.write_header()
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)

    def write_header(self):
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
        self._header_written = True

    def save_checkpoint(self, obj, name: str = "checkpoint.pth"):
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / name
        try:
            if hasattr(obj, 'save'):
                obj.save(str(path))
            else: # Fallback for objects without a .save method (e.g., dicts)
                with open(path.with_suffix('.json'), 'w') as f:
                    json.dump(obj, f, indent=4)
        except Exception as e:
            print(f"Failed to save checkpoint {name}: {e}")
            with open(path.with_suffix('.err.txt'), 'w') as f:
                f.write(f"Failed to save checkpoint: {e}")
        return str(path)

    def get_run_dir(self) -> str:
        return str(self.run_dir)