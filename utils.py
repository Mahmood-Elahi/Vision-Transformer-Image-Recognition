# ============================================================
# utils.py
# ------------------------------------------------------------
# This file contains helper functions for:
# - Reproducibility
# - Folder creation
# ============================================================

import os
import random
import numpy as np
import torch

from config import SEED, CHECKPOINT_DIR, RESULTS_DIR


def set_seed(seed=SEED):
    """
    Sets random seeds for reproducibility.

    This helps make results more consistent across runs.
    It does not guarantee perfectly identical GPU results, but it
    reduces randomness significantly.
    """

    # Python random module.
    random.seed(seed)

    # NumPy random module.
    np.random.seed(seed)

    # PyTorch CPU random seed.
    torch.manual_seed(seed)

    # PyTorch CUDA random seed for all GPUs.
    torch.cuda.manual_seed_all(seed)


def create_dirs():
    """
    Creates folders needed for checkpoints and result outputs.
    """

    # Folder for saved model checkpoints.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Folder for result JSON files.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Folder for saved graphs.
    os.makedirs(f"{RESULTS_DIR}/graphs", exist_ok=True)