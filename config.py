# ============================================================
# config.py
# ------------------------------------------------------------
# This file stores all global settings for the CIFAR-10
# Vision Transformer project.
#
# Keeping settings here makes it easy to tune the model without
# editing multiple files.
# ============================================================


# Random seed used for reproducibility.
# This makes dataset splitting, weight initialization, and random
# operations more consistent across runs.
SEED = 42


# ============================================================
# Training settings
# ============================================================

# Number of images processed at once.
# Larger batch size can improve GPU efficiency, but uses more VRAM.
BATCH_SIZE = 512

# Maximum number of training epochs.
# One epoch = one full pass through the training dataset.
EPOCHS = 150

# Stops training if validation accuracy does not improve for this
# many consecutive epochs.
EARLY_STOPPING_PATIENCE = 25


# ============================================================
# Dataset / image settings
# ============================================================

# CIFAR-10 images are 32x32 pixels.
IMAGE_SIZE = 32

# Patch size controls how the image is split into tokens.
# PATCH_SIZE = 4 means each 32x32 image becomes 8x8 = 64 patches.
# Smaller patch size = more detail but slower training.
PATCH_SIZE = 4

# CIFAR-10 images are RGB, so there are 3 input channels.
IN_CHANNELS = 3

# CIFAR-10 has 10 classes.
NUM_CLASSES = 10


# ============================================================
# Vision Transformer architecture settings
# ============================================================

# Embedding dimension for each image token.
# Higher values increase model capacity but also increase compute.
EMBED_DIM = 384

# Number of transformer blocks.
# Deeper models can learn more complex patterns but train slower.
DEPTH = 12

# Number of attention heads in each transformer block.
NUM_HEADS = 6

# Hidden dimension inside the MLP section of each transformer block.
MLP_DIM = 1536


# ============================================================
# Regularization settings
# ============================================================

# Dropout randomly removes activations during training.
# This helps prevent overfitting.
DROPOUT = 0.10

# Drop path randomly skips residual paths during training.
# This is common in modern transformer training.
DROP_PATH_RATE = 0.10


# ============================================================
# Optimizer / learning-rate scheduler settings
# ============================================================

# Peak learning rate after warmup.
LEARNING_RATE = 5e-4

# Minimum learning rate reached near the end of cosine decay.
MIN_LR = 1e-6

# Weight decay penalizes large weights and helps generalization.
WEIGHT_DECAY = 0.05

# Number of epochs used to slowly increase the learning rate.
# Warmup stabilizes transformer training.
WARMUP_EPOCHS = 15


# ============================================================
# Loss / augmentation settings
# ============================================================

# Label smoothing prevents the model from becoming too confident.
LABEL_SMOOTHING = 0.05

# Mixup blends two images and their labels together.
# Lower value = less aggressive mixing.
MIXUP_ALPHA = 0.4

# CutMix cuts a region from one image and pastes it into another.
CUTMIX_ALPHA = 1.0

# Probability of using either Mixup or CutMix on a batch.
MIX_PROB = 1.0

# Probability that the selected mixing method is CutMix.
# If CutMix is not selected, Mixup is used.
CUTMIX_PROB = 0.5


# ============================================================
# EMA settings
# ============================================================

# EMA = Exponential Moving Average.
# It keeps a smoothed copy of the model weights for evaluation.
EMA_DECAY = 0.999


# ============================================================
# System settings
# ============================================================

# Number of CPU workers used to load data.
# Higher values can improve speed if the CPU can keep up.
NUM_WORKERS = 8


# ============================================================
# File/folder paths
# ============================================================

# Folder where CIFAR-10 will be downloaded.
DATA_DIR = "./data"

# Folder where the best model checkpoint will be saved.
CHECKPOINT_DIR = "./checkpoints"

# Folder where graphs and result JSON files will be saved.
RESULTS_DIR = "./results"


# ============================================================
# CIFAR-10 class names
# ============================================================

# These labels are used for classification reports and confusion matrices.
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]