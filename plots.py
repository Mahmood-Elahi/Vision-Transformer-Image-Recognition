# ============================================================
# plots.py
# ------------------------------------------------------------
# This file creates and saves training graphs:
# - Loss curve
# - Validation accuracy curve
# - Learning-rate curve
# - Confusion matrix
# ============================================================

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from config import CIFAR10_CLASSES


def plot_single(values, title, ylabel, save_path):
    """
    Plots a single list of values over epochs.

    Used for:
    - Learning rate
    - Validation accuracy
    """

    # Create figure.
    plt.figure(figsize=(8, 6))

    # Plot values.
    plt.plot(values)

    # Add title and axis labels.
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    # Add grid for readability.
    plt.grid(True)

    # Save graph to file.
    plt.savefig(save_path)

    # Close figure to free memory.
    plt.close()


def plot_train_val(train_values, val_values, title, ylabel, save_path):
    """
    Plots training and validation values on the same graph.

    Used for:
    - Training loss vs validation loss
    """

    # Create figure.
    plt.figure(figsize=(8, 6))

    # Plot training and validation curves.
    plt.plot(train_values, label="Train")
    plt.plot(val_values, label="Validation")

    # Add title and axis labels.
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    # Add legend to identify each line.
    plt.legend()

    # Add grid for readability.
    plt.grid(True)

    # Save graph to file.
    plt.savefig(save_path)

    # Close figure to free memory.
    plt.close()


def save_confusion_matrix(cm, save_path):
    """
    Saves a confusion matrix plot.

    Confusion matrix shows:
    - Rows: true labels
    - Columns: predicted labels
    """

    # Create sklearn confusion matrix display object.
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CIFAR10_CLASSES,
    )

    # Create larger figure because CIFAR-10 has 10 class names.
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot confusion matrix with rotated x-axis labels.
    display.plot(ax=ax, xticks_rotation=45)

    # Adjust layout so labels do not overlap.
    plt.tight_layout()

    # Save plot.
    plt.savefig(save_path)

    # Close figure to free memory.
    plt.close()