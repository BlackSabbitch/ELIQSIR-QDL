# utils.py

import math
import inspect
from typing import Dict, Any, Callable
from logger import *
from datetime import datetime
import os
import json


class Utils:
    """
    Utility functions for various helper operations.

    Provides static methods for loss visualization, argument filtering,
    and other common utilities used throughout the project.
    """

    @staticmethod
    def get_loss_bar(loss_val: float, bar_len: int = 10) -> str:
        """
        Generate a visual bar representation of loss value.

        Uses logarithmic scaling to handle wide dynamic range of loss values.
        Returns a string bar with filled blocks representing loss magnitude.

        Args:
            loss_val: Loss value to visualize.
            bar_len: Length of the bar in characters.

        Returns:
            String representation of the loss bar.

        Example:
            >>> bar = Utils.get_loss_bar(0.5, bar_len=10)
            >>> print(bar)  # [████████  ]
        """
        # Use log10 for normalizing dynamic range
        # Add 1e-9 to avoid log(0)
        log_loss = math.log10(loss_val + 1e-9)
        
        # Scale: assume loss from 50 (log ~1.7) to 0.1 (log -1)
        # Simple linear coloring for log_loss range from -1 to 2
        level = (log_loss + 1) / 3  # normalize to [0, 1]
        level = max(0, min(1, level))  # clamp
        
        filled = int(level * bar_len)
        return "[" + "█" * filled + " " * (bar_len - filled) + "]"

    @staticmethod
    def filter_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter keyword arguments to only those accepted by the function.

        Inspects the function signature and returns only the arguments
        that the function can accept.

        Args:
            func: Function or class to inspect.
            kwargs: Dictionary of keyword arguments.

        Returns:
            Filtered dictionary containing only valid arguments.

        Example:
            >>> def my_func(a, b=10):
            ...     return a + b
            >>> kwargs = {'a': 1, 'b': 2, 'c': 3}
            >>> filtered = Utils.filter_kwargs(my_func, kwargs)
            >>> print(filtered)  # {'a': 1, 'b': 2}
        """
        sig = inspect.signature(func)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    @staticmethod
    def handle_metadata(config, train_ds, val_ds, test_ds):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"runs/{config['experiment_name']}_{timestamp}"
        os.makedirs(exp_dir, exist_ok=True)
        print(f"!!! ФАЙЛЫ СОХРАНЯЮТСЯ СЮДА: {exp_dir} !!!")
        config['dataset']['actual_sizes'] = {
            'train': len(train_ds),
            'val': len(val_ds),
            'test': len(test_ds)
        }
        with open(f"{exp_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=4)

        return exp_dir
