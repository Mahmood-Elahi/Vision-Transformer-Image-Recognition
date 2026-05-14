# ============================================================
# main.py
# ------------------------------------------------------------
# This is the main entry point for training the CIFAR-10
# Vision Transformer model.
#
# It handles:
# - Setup
# - Model creation
# - Training loop
# - Validation
# - Early stopping
# - Checkpoint saving
# - Final test evaluation
# - Graph/result saving
# ============================================================

import json
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    IMAGE_SIZE,
    PATCH_SIZE,
    IN_CHANNELS,
    NUM_CLASSES,
    EMBED_DIM,
    DEPTH,
    NUM_HEADS,
    MLP_DIM,
    DROPOUT,
    DROP_PATH_RATE,
    LEARNING_RATE,
    MIN_LR,
    WEIGHT_DECAY,
    WARMUP_EPOCHS,
    LABEL_SMOOTHING,
    MIXUP_ALPHA,
    CUTMIX_ALPHA,
    EMA_DECAY,
    CHECKPOINT_DIR,
    RESULTS_DIR,
)

from utils import set_seed, create_dirs
from data import get_dataloaders
from models import VisionTransformer, ModelEMA
from scheduler import get_lr, set_optimizer_lr
from engine import train_one_epoch, validate, evaluate_model
from plots import plot_train_val, plot_single, save_confusion_matrix


def main():
    """
    Main training function.
    """

    # ========================================================
    # Setup
    # ========================================================

    # Set seeds for reproducibility.
    set_seed()

    # Create required folders.
    create_dirs()

    # Select GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    if device.type == "cuda":
        # Print GPU name.
        print("GPU:", torch.cuda.get_device_name(0))

        # Enables cuDNN auto-tuning for faster fixed-size image training.
        torch.backends.cudnn.benchmark = True

        # Improves matrix multiplication performance on supported GPUs.
        torch.set_float32_matmul_precision("high")

    else:
        print("WARNING: CPU training will be extremely slow.")

    # ========================================================
    # Data
    # ========================================================

    # Load train, validation, and test DataLoaders.
    train_loader, val_loader, test_loader = get_dataloaders()

    # ========================================================
    # Model
    # ========================================================

    # Create Vision Transformer model.
    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
        drop_path_rate=DROP_PATH_RATE,
    ).to(device)

    # Create EMA version of the model.
    # EMA model is used for validation and final testing.
    ema_model = ModelEMA(model, decay=EMA_DECAY)

    # Cross-entropy loss for validation/testing.
    # Training uses soft-target loss because Mixup/CutMix creates soft labels.
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # AdamW is commonly used for transformer training.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # GradScaler supports stable mixed precision training.
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(device.type == "cuda"),
    )

    # ========================================================
    # Tracking variables
    # ========================================================

    # Best validation accuracy seen so far.
    best_val_acc = 0.0

    # Best validation loss associated with best validation accuracy.
    best_val_loss = float("inf")

    # Epoch where best validation accuracy occurred.
    best_epoch = 0

    # Counts epochs without validation accuracy improvement.
    epochs_without_improvement = 0

    # Path where the best model will be saved.
    best_path = f"{CHECKPOINT_DIR}/best_pure_vit_cifar10.pth"

    # Lists for plotting.
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []

    # Track total training time.
    total_start = time.time()

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(EPOCHS):
        # Track epoch time.
        epoch_start = time.time()

        # Get learning rate for this epoch.
        lr = get_lr(epoch)

        # Apply learning rate to optimizer.
        set_optimizer_lr(optimizer, lr)

        # Save learning rate for plotting.
        learning_rates.append(lr)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Learning Rate: {lr:.8f}")

        # Train model for one epoch.
        train_loss = train_one_epoch(
            model=model,
            ema_model=ema_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        # Validate EMA model.
        val_loss, val_acc = validate(
            model=ema_model.ema,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Store metrics for graphs.
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Calculate epoch duration.
        epoch_time = time.time() - epoch_start

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Epoch Time: {epoch_time / 60:.2f} min"
        )

        # ====================================================
        # Best model checkpointing
        # ====================================================

        # Save model only when validation accuracy improves.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # Reset early-stopping counter.
            epochs_without_improvement = 0

            # Save model checkpoint.
            torch.save(
                {
                    "model_state_dict": ema_model.ema.state_dict(),
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "epoch": best_epoch,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "image_size": IMAGE_SIZE,
                        "patch_size": PATCH_SIZE,
                        "embed_dim": EMBED_DIM,
                        "depth": DEPTH,
                        "num_heads": NUM_HEADS,
                        "mlp_dim": MLP_DIM,
                        "dropout": DROPOUT,
                        "drop_path_rate": DROP_PATH_RATE,
                        "learning_rate": LEARNING_RATE,
                        "min_lr": MIN_LR,
                        "weight_decay": WEIGHT_DECAY,
                        "warmup_epochs": WARMUP_EPOCHS,
                        "label_smoothing": LABEL_SMOOTHING,
                        "mixup_alpha": MIXUP_ALPHA,
                        "cutmix_alpha": CUTMIX_ALPHA,
                        "ema_decay": EMA_DECAY,
                        "pretrained": False,
                        "model_type": "pure_vision_transformer",
                    },
                },
                best_path,
            )

            print("Saved new best EMA model.")

        else:
            # Increase early-stopping counter if validation accuracy did not improve.
            epochs_without_improvement += 1

            print(
                f"No validation improvement for "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs."
            )

            # Stop training if the model has not improved for too long.
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # ========================================================
    # Load best model
    # ========================================================

    # Load checkpoint with best validation accuracy.
    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    # Restore EMA model weights.
    ema_model.ema.load_state_dict(checkpoint["model_state_dict"])

    # ========================================================
    # Final test evaluation
    # ========================================================

    # Evaluate best EMA model on test set.
    cm, report, test_acc = evaluate_model(
        model=ema_model.ema,
        dataloader=test_loader,
        device=device,
    )

    # Calculate total runtime.
    total_minutes = (time.time() - total_start) / 60

    # ========================================================
    # Save graphs
    # ========================================================

    # Save training vs validation loss graph.
    plot_train_val(
        train_losses,
        val_losses,
        "Training vs Validation Loss",
        "Loss",
        f"{RESULTS_DIR}/graphs/loss_curve_pure_vit.png",
    )

    # Save learning-rate schedule graph.
    plot_single(
        learning_rates,
        "Learning Rate Schedule",
        "Learning Rate",
        f"{RESULTS_DIR}/graphs/learning_rate_pure_vit.png",
    )

    # Save validation accuracy graph.
    plot_single(
        val_accuracies,
        "Validation Accuracy",
        "Accuracy",
        f"{RESULTS_DIR}/graphs/validation_accuracy_pure_vit.png",
    )

    # Save confusion matrix graph.
    save_confusion_matrix(
        cm,
        f"{RESULTS_DIR}/graphs/confusion_matrix_pure_vit.png",
    )

    # ========================================================
    # Save results to JSON
    # ========================================================

    results = {
        "best_validation_accuracy": best_val_acc,
        "best_validation_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_accuracy": test_acc,
        "classification_report": report,
        "total_training_minutes": total_minutes,
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "depth": DEPTH,
            "num_heads": NUM_HEADS,
            "mlp_dim": MLP_DIM,
            "dropout": DROPOUT,
            "drop_path_rate": DROP_PATH_RATE,
            "learning_rate": LEARNING_RATE,
            "min_lr": MIN_LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS,
            "label_smoothing": LABEL_SMOOTHING,
            "mixup_alpha": MIXUP_ALPHA,
            "cutmix_alpha": CUTMIX_ALPHA,
            "ema_decay": EMA_DECAY,
            "pretrained": False,
            "model_type": "pure_vision_transformer",
        },
    }

    # Write results to JSON file.
    with open(f"{RESULTS_DIR}/pure_vit_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # ========================================================
    # Print final summary
    # ========================================================

    print("\n================ FINAL RESULTS ================")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Total Training Time: {total_minutes:.2f} minutes")
    print("\nClassification Report:")
    print(report)
    print("================================================")


# Run main function only when this file is executed directly.
if __name__ == "__main__":
    main()