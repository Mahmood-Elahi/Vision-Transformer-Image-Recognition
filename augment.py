# ============================================================
# augment.py
# ------------------------------------------------------------
# This file contains Mixup, CutMix, label smoothing, and soft
# target loss functions.
#
# These methods improve generalization by making the training
# task harder and reducing overconfidence.
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    NUM_CLASSES,
    LABEL_SMOOTHING,
    MIXUP_ALPHA,
    CUTMIX_ALPHA,
    MIX_PROB,
    CUTMIX_PROB,
)


def smooth_one_hot(labels, num_classes=NUM_CLASSES, smoothing=LABEL_SMOOTHING):
    """
    Converts class labels into smoothed one-hot vectors.

    Normal one-hot example for class 3:
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    Smoothed version:
    [0.005, 0.005, 0.005, 0.955, 0.005, ...]

    This prevents the model from becoming too confident.
    """

    confidence = 1.0 - smoothing
    smoothing_value = smoothing / num_classes

    # Create a tensor filled with the smoothing value.
    targets = torch.full(
        size=(labels.size(0), num_classes),
        fill_value=smoothing_value,
        device=labels.device,
    )

    # Place the confidence value at the correct class index.
    targets.scatter_(1, labels.unsqueeze(1), confidence)

    return targets


def rand_bbox(size, lam):
    """
    Creates a random rectangular box for CutMix.

    The box determines which part of one image will be replaced
    by a patch from another image.
    """

    # Image width and height.
    width = size[2]
    height = size[3]

    # Cut ratio controls how large the CutMix box is.
    cut_ratio = np.sqrt(1.0 - lam)

    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    # Random center point for the box.
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    # Calculate box boundaries and keep them inside the image.
    bbx1 = np.clip(cx - cut_width // 2, 0, width)
    bby1 = np.clip(cy - cut_height // 2, 0, height)
    bbx2 = np.clip(cx + cut_width // 2, 0, width)
    bby2 = np.clip(cy + cut_height // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


def apply_mixup_cutmix(images, labels):
    """
    Applies Mixup or CutMix to a batch of images.

    Mixup:
    - Blends two full images together.
    - Blends their labels proportionally.

    CutMix:
    - Cuts a patch from one image.
    - Pastes it into another image.
    - Blends labels based on patch area.
    """

    # If augmentation is skipped, return original images with smoothed labels.
    if np.random.rand() > MIX_PROB:
        targets = smooth_one_hot(labels)
        return images, targets

    # Convert labels to smoothed one-hot vectors.
    targets = smooth_one_hot(labels)

    # Number of images in the batch.
    batch_size = images.size(0)

    # Randomly shuffle image indices so each image can be mixed with another.
    index = torch.randperm(batch_size, device=images.device)

    # Decide whether to use CutMix or Mixup.
    use_cutmix = np.random.rand() < CUTMIX_PROB

    # ========================================================
    # CutMix branch
    # ========================================================

    if use_cutmix and CUTMIX_ALPHA > 0:
        # Sample lambda from beta distribution.
        lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)

        # Create random bounding box.
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

        # Clone images so the original batch is not directly modified.
        mixed_images = images.clone()

        # Replace the selected rectangular region with pixels
        # from another randomly selected image.
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[
            index,
            :,
            bbx1:bbx2,
            bby1:bby2,
        ]

        # Recalculate lambda based on the actual pasted area.
        area = (bbx2 - bbx1) * (bby2 - bby1)
        lam = 1.0 - area / (images.size(2) * images.size(3))

        # Blend the labels according to the remaining image area.
        mixed_targets = lam * targets + (1.0 - lam) * targets[index]

        return mixed_images, mixed_targets

    # ========================================================
    # Mixup branch
    # ========================================================

    if MIXUP_ALPHA > 0:
        # Sample lambda from beta distribution.
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)

        # Blend the images.
        mixed_images = lam * images + (1.0 - lam) * images[index]

        # Blend the labels.
        mixed_targets = lam * targets + (1.0 - lam) * targets[index]

        return mixed_images, mixed_targets

    # Fallback: return original images and smoothed labels.
    return images, targets


def soft_target_cross_entropy(outputs, targets):
    """
    Cross-entropy loss for soft labels.

    Standard CrossEntropyLoss expects hard labels like:
    class = 3

    Mixup/CutMix creates soft labels like:
    70% cat, 30% dog

    This function supports those soft label targets.
    """

    # Convert logits to log probabilities.
    log_probs = F.log_softmax(outputs, dim=1)

    # Compute cross entropy manually for soft targets.
    loss = -(targets * log_probs).sum(dim=1).mean()

    return loss