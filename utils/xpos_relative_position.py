# Copyright (c) 2022 Microsoft
# Licensed under The MIT License (https://github.com/microsoft/torchscale/blob/main/LICENSE)

import torch
import torch.nn as nn


def fixed_pos_embedding(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate fixed position embeddings as sinusoidal functions.
    
    Args:
        x: Tensor of shape (seq_len, dim) containing position scales
        
    Returns:
        Tuple of (sin, cos) tensors of shape (seq_len, dim)
    """
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = torch.einsum("i,j->ij", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate elements in each pair of consecutive dimensions.
    
    Args:
        x: Tensor of shape (..., d) where d is divisible by 2
        
    Returns:
        Tensor with rotated pairs of elements
    """
    x1 = x[:, :, :, ::2]    # Get even indices: 0, 2, 4...
    x2 = x[:, :, :, 1::2]   # Get odd indices: 1, 3, 5...
    x = torch.stack((-x2, x1), dim=-1)  # Stack with negative of odds first
    return x.flatten(-2)  # Flatten last two dimensions


def duplicate_interleave(m: torch.Tensor) -> torch.Tensor:
    """
    Duplicate a matrix and interleave the copy.
    
    A simple version of `torch.repeat_interleave` for duplicating a matrix
    while interleaving the copy.
    
    Args:
        m: Input tensor of shape (dim0, dim1)
        
    Returns:
        Tensor with interleaved duplicated values
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)        # Flatten the matrix
    m = m.repeat(1, 2)       # Repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)     # Reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, 
                         scale: float = 1.0) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor.
    
    Args:
        x: Input tensor of shape (batch_size, n_heads, seq_len, head_dim)
        sin: Sine tensor from positional embedding
        cos: Cosine tensor from positional embedding
        scale: Additional scaling factor for the embeddings
        
    Returns:
        Tensor with rotary positional embeddings applied
    """
    # Duplicate and interleave sine and cosine with specified scale
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    
    # Apply rotation formula: (x * cos) + (rotate_every_two(x) * sin)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    """
    eXponential POSitional encoding for transformer models.
    
    This implements position encoding where the influence of position decreases
    exponentially with distance. The scaling helps with extrapolation to longer
    sequences than seen during training.
    
    References:
        - Sun et al., "Length-Extrapolatable Transformers", 2022
        - https://github.com/microsoft/torchscale
    """
    def __init__(self, head_dim: int, scale_base: int = 512):
        """
        Initialize XPOS module.
        
        Args:
            head_dim: Dimension of each attention head
            scale_base: Base value for scaling positional weights
        """
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        
        # Register scaling factors as a buffer (persistent but not model parameters)
        self.register_buffer(
            "scale", 
            (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x: torch.Tensor, offset: int = 0, downscale: bool = False) -> torch.Tensor:
        """
        Apply XPOS to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, n_heads, seq_len, head_dim)
            offset: Position offset to apply
            downscale: If True, apply downscaling (inverse) of positional weights
            
        Returns:
            Tensor with positional encodings applied
        """
        length = x.shape[2]  # Sequence length is the third dimension
        min_pos = 0
        max_pos = length + offset + min_pos
        
        # Calculate exponential scaling factors for each position
        pos_scales = torch.arange(min_pos, max_pos, 1).to(self.scale)
        scale = self.scale ** pos_scales.div(self.scale_base)[:, None]
        
        # Generate sinusoidal position embeddings
        sin, cos = fixed_pos_embedding(scale)

        # Handle case where calculated positions exceed sequence length
        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        # Apply inverse scaling if downscale is True
        if downscale:
            scale = 1 / scale

        # Apply rotary position embeddings with scaling
        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
    
    def forward_reverse(self, x: torch.Tensor, offset: int = 0, downscale: bool = False) -> torch.Tensor:
        """
        Apply XPOS with reversed sinusoidal components for inverse operations.
        
        Args:
            x: Input tensor of shape (batch_size, n_heads, seq_len, head_dim)
            offset: Position offset to apply
            downscale: If True, apply downscaling (inverse) of positional weights
            
        Returns:
            Tensor with reversed positional encodings applied
        """
        length = x.shape[2]  # Sequence length is the third dimension
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        
        # Calculate exponential scaling factors for each position
        pos_scales = torch.arange(min_pos, max_pos, 1).to(self.scale)
        scale = self.scale ** pos_scales.div(self.scale_base)[:, None]
        
        # Generate sinusoidal position embeddings
        sin, cos = fixed_pos_embedding(scale)

        # Handle case where calculated positions exceed sequence length
        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        # Apply inverse scaling if downscale is True
        if downscale:
            scale = 1 / scale

        # Apply rotary position embeddings with negated sine for reversal
        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x


# Test code
if __name__ == "__main__":
    # Create a test tensor (batch=1, heads=1, seq_len=4, head_dim=4)
    x = torch.eye(4).unsqueeze(0).unsqueeze(0)
    
    # Initialize XPOS with head dimension 4
    xpos = XPOS(4)
    
    # Apply XPOS encoding
    x_rot = xpos(x)
    
    # Apply reverse XPOS encoding
    x_rot_rev = xpos.forward_reverse(x)
    
    # Check if forward and reverse operations are approximately inverse
    # by multiplying the forward result with the transpose of the reverse result
    similarity = x_rot @ x_rot_rev.transpose(-1, -2)
    print("Similarity matrix (should be close to identity):")
    print(similarity.squeeze())