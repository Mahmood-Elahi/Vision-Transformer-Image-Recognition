# ============================================================
# scheduler.py
# ------------------------------------------------------------
# This file controls the learning-rate schedule.
#
# The schedule uses:
# 1. Linear warmup
# 2. Cosine decay
# ============================================================

import math

from config import EPOCHS, WARMUP_EPOCHS, LEARNING_RATE, MIN_LR


def get_lr(epoch):
    """
    Returns the learning rate for a given epoch.

    During warmup:
    - LR slowly increases from small value to LEARNING_RATE.

    After warmup:
    - LR follows cosine decay toward MIN_LR.
    """

    # ========================================================
    # Warmup phase
    # ========================================================

    if epoch < WARMUP_EPOCHS:
        return LEARNING_RATE * float(epoch + 1) / float(WARMUP_EPOCHS)

    # ========================================================
    # Cosine decay phase
    # ========================================================

    # Progress ranges from 0 to 1 after warmup.
    progress = float(epoch - WARMUP_EPOCHS) / float(EPOCHS - WARMUP_EPOCHS)

    # Clamp progress so it stays between 0 and 1.
    progress = min(1.0, max(0.0, progress))

    # Cosine decay starts near 1 and smoothly decreases to 0.
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    # Scale cosine decay between LEARNING_RATE and MIN_LR.
    return MIN_LR + (LEARNING_RATE - MIN_LR) * cosine_decay


def set_optimizer_lr(optimizer, lr):
    """
    Updates the learning rate for all optimizer parameter groups.
    """

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr