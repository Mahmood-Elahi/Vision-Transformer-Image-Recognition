# ============================================================
# data.py
# ------------------------------------------------------------
# This file prepares the CIFAR-10 dataset and creates PyTorch
# DataLoaders for training, validation, and testing.
# ============================================================

import numpy as np
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset

from config import (
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)


def get_dataloaders():
    """
    Creates and returns the train, validation, and test DataLoaders.

    CIFAR-10 originally has:
    - 50,000 training images
    - 10,000 test images

    This code splits the 50,000 training images into:
    - 45,000 training images
    - 5,000 validation images

    The test set remains separate and is only used at the end.
    """

    # ========================================================
    # Training transforms
    # --------------------------------------------------------
    # These transformations are applied only to training images.
    # They help the model generalize by showing it slightly
    # different versions of the same images.
    # ========================================================

    train_transform = transforms.Compose([
        # Randomly crops the image after padding it by 4 pixels.
        # This teaches the model to handle small position shifts.
        transforms.RandomCrop(32, padding=4),

        # Randomly flips images horizontally.
        # This is useful because objects can appear facing either direction.
        transforms.RandomHorizontalFlip(),

        # AutoAugment applies a learned augmentation policy designed for CIFAR-10.
        # This was used in the higher-accuracy version.
        transforms.AutoAugment(
            policy=transforms.AutoAugmentPolicy.CIFAR10,
        ),

        # Converts PIL image to PyTorch tensor.
        transforms.ToTensor(),

        # Normalizes CIFAR-10 images using dataset mean and standard deviation.
        # This makes optimization more stable.
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),

        # RandomErasing removes a random rectangle from the image.
        # This prevents the model from relying too much on one small region.
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.20),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])

    # ========================================================
    # Evaluation transforms
    # --------------------------------------------------------
    # Validation and test images should not use random
    # augmentation. They should only be converted and normalized.
    # ========================================================

    eval_transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    # ========================================================
    # Load CIFAR-10 datasets
    # ========================================================

    # Training dataset with augmentation.
    train_full_aug = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Same training dataset, but without augmentation.
    # This is used for validation so validation images are clean.
    train_full_eval = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=eval_transform,
    )

    # Official CIFAR-10 test dataset.
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=eval_transform,
    )

    # ========================================================
    # Create train/validation split
    # ========================================================

    # Create an array of indices from 0 to 49,999.
    indices = np.arange(len(train_full_aug))

    # Shuffle indices using the fixed seed for reproducibility.
    np.random.seed(SEED)
    np.random.shuffle(indices)

    # First 45,000 images are used for training.
    train_indices = indices[:45000]

    # Last 5,000 images are used for validation.
    val_indices = indices[45000:]

    # Subset applies the selected indices to each dataset.
    train_dataset = Subset(train_full_aug, train_indices)
    val_dataset = Subset(train_full_eval, val_indices)

    # Persistent workers keep DataLoader workers alive between epochs.
    # This reduces overhead and can improve training speed.
    persistent_workers = NUM_WORKERS > 0

    # ========================================================
    # Create DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader