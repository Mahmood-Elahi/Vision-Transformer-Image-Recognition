# ============================================================
# engine.py
# ------------------------------------------------------------
# This file contains the main training, validation, and testing
# loops for the Vision Transformer.
# ============================================================

import numpy as np
import torch

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from config import CIFAR10_CLASSES
from augment import apply_mixup_cutmix, soft_target_cross_entropy


def train_one_epoch(model, ema_model, dataloader, optimizer, scaler, device):
    """
    Trains the model for one epoch.

    Steps:
    1. Load a batch of images and labels.
    2. Apply Mixup/CutMix.
    3. Run forward pass.
    4. Compute soft-label loss.
    5. Backpropagate gradients.
    6. Clip gradients.
    7. Update model weights.
    8. Update EMA model.
    """

    # Set model to training mode.
    model.train()

    # Track total loss across all batches.
    total_loss = 0.0

    # Loop through training batches.
    for images, labels in tqdm(dataloader, desc="Training"):
        # Move images and labels to GPU/CPU.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply Mixup or CutMix augmentation.
        mixed_images, mixed_targets = apply_mixup_cutmix(images, labels)

        # Clear previous gradients.
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision autocast improves GPU speed and reduces memory use.
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            # Forward pass.
            outputs = model(mixed_images)

            # Loss for soft labels from Mixup/CutMix.
            loss = soft_target_cross_entropy(outputs, mixed_targets)

        # Scale loss for stable mixed precision training.
        scaler.scale(loss).backward()

        # Unscale gradients before clipping.
        scaler.unscale_(optimizer)

        # Clip gradients to avoid exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model weights.
        scaler.step(optimizer)

        # Update GradScaler.
        scaler.update()

        # Update EMA model after optimizer step.
        ema_model.update(model)

        # Add current batch loss.
        total_loss += loss.item()

    # Average loss across all batches.
    avg_loss = total_loss / len(dataloader)

    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Validation does not use Mixup/CutMix.
    Validation measures clean-image performance.
    """

    # Set model to evaluation mode.
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculations during validation.
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            # Move batch to GPU/CPU.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixed precision inference.
            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                # Forward pass.
                outputs = model(images)

                # Standard cross-entropy because validation labels are hard labels.
                loss = criterion(outputs, labels)

            # Add batch loss.
            total_loss += loss.item()

            # Choose class with highest logit.
            predictions = outputs.argmax(dim=1)

            # Count correct predictions.
            correct += (predictions == labels).sum().item()

            # Count total labels.
            total += labels.size(0)

    # Average validation loss.
    avg_loss = total_loss / len(dataloader)

    # Validation accuracy.
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_model(model, dataloader, device):
    """
    Evaluates the final best model on the test set.

    Returns:
    - Confusion matrix
    - Classification report
    - Test accuracy
    """

    # Set model to evaluation mode.
    model.eval()

    all_predictions = []
    all_labels = []

    # Disable gradients during testing.
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            # Move images to GPU/CPU.
            images = images.to(device, non_blocking=True)

            # Mixed precision inference.
            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                # Forward pass.
                outputs = model(images)

            # Convert logits to predicted class indices.
            predictions = outputs.argmax(dim=1).cpu()

            # Store predictions and labels for metrics.
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

    # Compute confusion matrix.
    cm = confusion_matrix(all_labels, all_predictions)

    # Compute precision, recall, and F1 score for each class.
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CIFAR10_CLASSES,
    )

    # Compute overall test accuracy.
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    return cm, report, accuracy