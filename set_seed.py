'''Set random seed across all main libraries for reproducibility.'''

import random
import numpy as np
import torch


def set_random_seed(seed: int):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    # torch.cuda.manual_seed_all(seed)
