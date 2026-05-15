# Vision Transformer Image Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange.svg)
![Torchvision](https://img.shields.io/badge/Torchvision-0.26-red.svg)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)
![Model](https://img.shields.io/badge/Model-Vision%20Transformer-purple.svg)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%20A6000-76B900.svg)

This project uses a pure Vision Transformer built with PyTorch to classify images from the CIFAR-10 dataset.

The project focuses on a complete computer vision training pipeline with patch-based tokenization, transformer encoder blocks, data augmentation, validation-based checkpointing, mixed-precision training, EMA model averaging, GPU acceleration, and final test-set evaluation.

---

## Project Summary

```text
Dataset: CIFAR-10
Task: 10-class image classification
Model: Pure Vision Transformer
Best checkpoint metric: Highest validation accuracy
Training GPU: NVIDIA RTX A6000
Best test accuracy: 91.06%
```

The model does not use pretrained weights. It learns image representations directly from CIFAR-10.

---

## Dataset

CIFAR-10 contains 60,000 RGB images across 10 classes:

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

Each image has shape:

```text
3 × 32 × 32
```

Dataset split:

| Dataset | Images | Purpose |
|---|---:|---|
| Training | 45,000 | Updates model weights |
| Validation | 5,000 | Selects best checkpoint |
| Test | 10,000 | Final evaluation only |

The test set is not used during training or model selection.

---

## Model Architecture

The model uses a pure Vision Transformer structure:

```text
Input image: 3 × 32 × 32
↓
Patch embedding
↓
Patch tokens
↓
CLS token + positional embeddings
↓
Transformer encoder blocks
↓
Layer normalization
↓
Mean pooling
↓
Linear classifier
↓
10-class output
```

Main architecture settings:

| Setting | Value |
|---|---:|
| Patch size | 2 × 2 |
| Image tokens | 256 |
| Tokens with CLS | 257 |
| Embedding dimension | 384 |
| Transformer depth | 12 |
| Attention heads | 6 |
| MLP dimension | 1536 |

---

## Shape Flow

```text
Input:
[batch_size, 3, 32, 32]

Patch embedding:
[batch_size, 384, 16, 16]

Flatten patches:
[batch_size, 384, 256]

Transpose into tokens:
[batch_size, 256, 384]

Add CLS token:
[batch_size, 257, 384]

After transformer blocks:
[batch_size, 257, 384]

Mean pooling:
[batch_size, 384]

Final output:
[batch_size, 10]
```

---

## Key Design Choices

### Patch Size

This version uses patch size 2.

```text
32 × 32 image
2 × 2 patches
16 × 16 patch grid
256 image tokens
257 total tokens including CLS
```

A smaller patch size preserves more image detail, which helps accuracy. The tradeoff is slower training because self-attention becomes more expensive as token count increases.

### Positional Embeddings

Transformers do not naturally know spatial order. Positional embeddings are added so the model knows where each patch came from.

### Self-Attention

Self-attention allows each image patch token to compare itself with every other patch token. This helps the model learn relationships between image regions.

## Training Pipeline

```text
Load CIFAR-10 batch
↓
Apply data augmentation
↓
Apply Mixup or CutMix
↓
Split images into patch tokens
↓
Pass tokens through transformer blocks
↓
Generate 10 class scores
↓
Compute soft-label loss
↓
Backpropagate gradients
↓
Clip gradients
↓
Update weights with AdamW
↓
Update EMA model
```

During validation and testing, the model only performs prediction and evaluation. It does not update weights.

---

## Augmentation and Regularization

| Method | Purpose |
|---|---|
| Random crop | Handles small image shifts |
| Random horizontal flip | Improves orientation generalization |
| AutoAugment | Applies learned CIFAR-10 augmentation policy |
| Random erasing | Prevents reliance on small regions |
| Mixup | Blends images and labels |
| CutMix | Replaces part of one image with another |
| Drop path | Skips residual paths during training |
| Weight decay | Penalizes large weights |
| Label smoothing | Reduces overconfidence |
| EMA | Uses smoothed model weights for evaluation |
| Gradient clipping | Prevents unstable gradient spikes |

This version uses `DROPOUT = 0.0` because the model already uses strong regularization through drop path, label smoothing, Mixup, CutMix, AutoAugment, RandomErasing, and weight decay.

---

## Learning Rate Schedule

The project uses linear warmup followed by cosine decay.

| Setting | Value |
|---|---:|
| Peak learning rate | 5e-4 |
| Minimum learning rate | 1e-6 |
| Warmup epochs | 20 |
| Maximum epochs | 200 |
| Weight decay | 0.05 |

Warmup stabilizes early transformer training. Cosine decay lowers the learning rate gradually as training progresses.

---

## Best Model Selection

The model does not blindly use the final epoch.

Validation accuracy is checked after every epoch. If validation accuracy improves, the EMA model weights are saved as the best checkpoint.

```text
Best model metric: Highest validation accuracy
Best epoch: 184
Best validation accuracy: 90.98%
Final test accuracy: 91.06%
```

Final test evaluation is performed using the best validation-accuracy checkpoint.

---

## GPU Training

This project was trained with CUDA acceleration on an NVIDIA RTX A6000 GPU.

The code automatically checks for CUDA:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Training speed depends heavily on the hardware used. The reported training time was measured on an NVIDIA RTX A6000. Runtime may differ on other GPUs, CPUs, or cloud environments.

CPU-only training is not recommended because Vision Transformers are computationally expensive, especially with smaller patch sizes.

---

## Results

| Metric | Value |
|---|---:|
| Best Validation Accuracy | 90.98% |
| Best Validation Loss | 0.7326 |
| Best Epoch | 184 |
| Final Test Accuracy | 91.06% |
| Total Training Time | 213.83 minutes |

The final test accuracy of 91.06% shows that the selected Vision Transformer generalized well to unseen CIFAR-10 test images.

---

## Visual Results

### Validation Accuracy
![Validation Accuracy](results/graphs/validation_accuracy_pure_vit.png)

### Training and Validation Loss
![Training and Validation Loss](results/graphs/loss_curve_pure_vit.png)

### Learning Rate Schedule
![Learning Rate Schedule](results/graphs/learning_rate_pure_vit.png)

### Confusion Matrix
![Confusion Matrix](results/graphs/confusion_matrix_pure_vit.png)

---

## Visual Outputs

The project saves outputs inside:

```text
results/
```

| File | Description |
|---|---|
| `pure_vit_results.json` | Final metrics and configuration |
| `loss_curve_pure_vit.png` | Training and validation loss |
| `validation_accuracy_pure_vit.png` | Validation accuracy over epochs |
| `learning_rate_pure_vit.png` | Warmup and cosine LR schedule |
| `confusion_matrix_pure_vit.png` | Class-level prediction errors |

---

## Project Structure

```text
Vision-Transformer-Image-Classifier/
│
├── main.py
├── config.py
├── data.py
├── models.py
├── engine.py
├── augment.py
├── scheduler.py
├── plots.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebooks/
│   └── vit_cifar10_demo.ipynb
│
└── results/
    └── graphs/
        ├── loss_curve_pure_vit.png
        ├── validation_accuracy_pure_vit.png
        ├── learning_rate_pure_vit.png
        └── confusion_matrix_pure_vit.png
```

---

## File Descriptions

| File | Purpose |
|---|---|
| `main.py` | Runs training, validation, checkpointing, testing, and saving |
| `config.py` | Stores hyperparameters and file paths |
| `data.py` | Loads CIFAR-10 and creates DataLoaders |
| `models.py` | Defines the Vision Transformer, DropPath, and EMA model |
| `engine.py` | Contains training, validation, and testing loops |
| `augment.py` | Contains Mixup, CutMix, label smoothing, and soft-label loss |
| `scheduler.py` | Defines warmup and cosine learning-rate scheduling |
| `plots.py` | Saves loss, accuracy, LR, and confusion matrix plots |
| `utils.py` | Sets random seeds and creates output folders |

---


## Jupyter Notebook Demo

This repo includes an optional notebook: `notebooks/vit_cifar10_demo.ipynb`.

It is not the main training pipeline. It is used to display the model results, graphs, confusion matrix, and Vision Transformer shape flow.

The main training code remains in `main.py`, `models.py`, `data.py`, `engine.py`, and the other Python files.

The notebook makes the project easier to review without rerunning the full training process.

---


## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- tqdm

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/Vision-Transformer-Image-Recognition.git
cd Vision-Transformer-Image-Recognition
pip install torch torchvision numpy matplotlib scikit-learn tqdm
python main.py
```

The script automatically downloads CIFAR-10 if it is not already available.

---

## What I Learned

- How Vision Transformers process images using patch tokens.
- How patch size affects training speed and accuracy.
- How self-attention connects image regions.
- How validation accuracy can be used for checkpoint selection.
- How Mixup, CutMix, label smoothing, and AutoAugment improve generalization.
- How mixed-precision training and EMA improve GPU training stability.
- How to evaluate a model using accuracy, loss curves, classification reports, and confusion matrices.

---

## Future Improvements

- Run multiple random seeds and average results
- Compare against CNN and ResNet baselines
- Test pretrained Vision Transformer weights
