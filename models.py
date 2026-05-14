# ============================================================
# models.py
# ------------------------------------------------------------
# This file defines the Vision Transformer model, transformer
# blocks, stochastic depth / DropPath, and EMA model wrapper.
# ============================================================

import copy
import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    DropPath, also called stochastic depth.

    Instead of dropping individual neurons like Dropout, DropPath
    randomly drops an entire residual branch during training.

    This helps regularize deep transformer models.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # Do nothing if drop probability is 0 or the model is in eval mode.
        if self.drop_prob == 0.0 or not self.training:
            return x

        # Probability of keeping the path.
        keep_prob = 1.0 - self.drop_prob

        # Shape ensures one random decision per sample, broadcast across features.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        # Generate random tensor on same device and dtype as input.
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )

        # Convert values to 0 or 1.
        random_tensor.floor_()

        # Scale surviving paths so the expected output remains stable.
        return x.div(keep_prob) * random_tensor


class TransformerBlock(nn.Module):
    """
    One standard Vision Transformer encoder block.

    Each block contains:
    1. LayerNorm
    2. Multi-head self-attention
    3. Residual connection
    4. LayerNorm
    5. MLP feed-forward network
    6. Residual connection
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, drop_path):
        super().__init__()

        # Normalization before attention.
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention lets each token attend to every other token.
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # DropPath applied to attention residual branch.
        self.drop_path1 = DropPath(drop_path)

        # Normalization before MLP.
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward MLP block.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # DropPath applied to MLP residual branch.
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        # Normalize tokens before attention.
        x_norm = self.norm1(x)

        # Self-attention: query, key, and value all come from x_norm.
        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=False,
        )

        # Residual connection after attention.
        x = x + self.drop_path1(attn_out)

        # Normalize before MLP.
        x_norm = self.norm2(x)

        # Feed-forward transformation.
        mlp_out = self.mlp(x_norm)

        # Residual connection after MLP.
        x = x + self.drop_path2(mlp_out)

        return x


class VisionTransformer(nn.Module):
    """
    Pure Vision Transformer for CIFAR-10.

    The model:
    1. Splits image into patches using Conv2d.
    2. Converts patches into token embeddings.
    3. Adds a CLS token and positional embeddings.
    4. Passes tokens through transformer blocks.
    5. Uses mean pooling over patch tokens.
    6. Classifies into 10 CIFAR-10 classes.
    """

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_dim=1536,
        dropout=0.10,
        drop_path_rate=0.10,
    ):
        super().__init__()

        # Image size must divide evenly into patches.
        assert image_size % patch_size == 0

        # Embedding dimension must divide evenly across attention heads.
        assert embed_dim % num_heads == 0

        # Number of patches per row and total number of patches.
        num_patches_per_row = image_size // patch_size
        num_patches = num_patches_per_row * num_patches_per_row

        # Patch embedding layer.
        # This converts each image patch into an embedding vector.
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Learnable CLS token.
        # In this code, the CLS token is included, but final classification
        # uses mean pooling over patch tokens instead.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embedding tells the transformer where each
        # patch token is located in the image.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        # Dropout after adding positional embeddings.
        self.dropout = nn.Dropout(dropout)

        # Linearly increase DropPath rate across layers.
        # Earlier layers get less DropPath, later layers get more.
        drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()

        # Stack transformer blocks.
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    drop_path=drop_rates[i],
                )
                for i in range(depth)
            ]
        )

        # Final normalization.
        self.norm = nn.LayerNorm(embed_dim)

        # Final classifier maps embedding dimension to class logits.
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Initialize weights.
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes model weights.

        Transformer models are sensitive to initialization, so truncated
        normal initialization is commonly used.
        """

        # Initialize CLS token and positional embedding.
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize linear layers and LayerNorm layers.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape starts as:
        # [batch_size, channels, height, width]
        batch_size = x.shape[0]

        # Convert image into patch embeddings.
        # Output shape:
        # [batch_size, embed_dim, patches_per_row, patches_per_row]
        x = self.patch_embed(x)

        # Flatten spatial patch grid into token sequence.
        # Shape becomes:
        # [batch_size, embed_dim, num_patches]
        x = x.flatten(2)

        # Move embedding dimension to the end.
        # Shape becomes:
        # [batch_size, num_patches, embed_dim]
        x = x.transpose(1, 2)

        # Expand CLS token for each image in the batch.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Add CLS token to the beginning of the token sequence.
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings.
        x = x + self.pos_embed

        # Apply dropout.
        x = self.dropout(x)

        # Pass through transformer encoder blocks.
        x = self.blocks(x)

        # Final normalization.
        x = self.norm(x)

        # Remove CLS token and keep only patch tokens.
        patch_tokens = x[:, 1:]

        # Mean pool across all patch tokens.
        # This often works well for small-image ViTs.
        pooled = patch_tokens.mean(dim=1)

        # Final classification logits.
        logits = self.classifier(pooled)

        return logits


class ModelEMA:
    """
    Exponential Moving Average model.

    EMA keeps a smoothed version of model weights:
    ema_weight = decay * ema_weight + (1 - decay) * current_weight

    The EMA model is often more stable for validation and testing.
    """

    def __init__(self, model, decay=0.999):
        # Create a deep copy of the model for EMA.
        self.ema = copy.deepcopy(model).eval()

        # EMA smoothing factor.
        self.decay = decay

        # EMA model is not trained directly by gradients.
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """
        Updates EMA weights using the current model weights.
        """

        # Get state dictionaries for EMA model and current model.
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        # Update every parameter/buffer.
        for key in ema_state.keys():
            if ema_state[key].dtype.is_floating_point:
                # Smooth floating-point tensors.
                ema_state[key].mul_(self.decay).add_(
                    model_state[key],
                    alpha=1.0 - self.decay,
                )
            else:
                # Directly copy non-floating tensors.
                ema_state[key].copy_(model_state[key])