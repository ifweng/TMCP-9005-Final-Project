import os
import gc
import math
from pathlib import Path
from typing import ClassVar, Optional, Union, List, Dict, Any
import json
import random

import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from einops import rearrange

from fastai.vision.all import *
from functools import partial
from xpos_relative_position import XPOS


def exists(val):
    """Check if a value exists (is not None)."""
    return val is not None


def pad_at_dim(t, pad, dim=-1, value=0.):
    """Pad a tensor at a specific dimension."""
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


class DynamicPositionBias(nn.Module):
    """Dynamic position bias module for attention mechanisms."""
    def __init__(self, dim, *, heads, depth, log_distance=False, norm=False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        """Get the device of the module."""
        return next(self.parameters()).device

    def forward(self, i, j):
        """Calculate dynamic position bias.
        
        Args:
            i, j: Sequence lengths
            
        Returns:
            Positional bias matrix
        """
        assert i == j
        n, device = j, self.device

        # Get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device=device)
        context_arange = torch.arange(n, device=device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # Input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device=device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            # Log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # Get position biases
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias


class AlibiPositionalBias(nn.Module):
    """Alibi positional bias for attention mechanisms."""
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)
    
    def get_bias(self, i, j, device):
        """Calculate the alibi bias.
        
        Args:
            i, j: Sequence lengths
            device: Computation device
            
        Returns:
            Alibi bias matrix
        """
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        """Generate slopes for alibi positional bias.
        
        Args:
            heads: Number of attention heads
            
        Returns:
            List of slopes for each head
        """
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    @property
    def device(self):
        """Get the device of the module."""
        return next(self.buffers()).device

    def forward(self, i, j):
        """Apply alibi positional bias.
        
        Args:
            i, j: Sequence lengths
            
        Returns:
            Positional bias matrix
        """
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer('bias', bias, persistent=False)

        return self.bias


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module."""
    def __init__(
        self,
        hidden_dim: int,
        positional_embedding: str,
        num_heads: Optional[int] = None,
        dropout: float = 0.10, 
        bias: bool = True,
        temperature: float = 1,
        use_se: bool = False,
    ):
        super().__init__()
        
        assert positional_embedding in ("xpos", "dyn", "alibi")
        self.positional_embedding = positional_embedding
        self.hidden_dim = hidden_dim
        if num_heads is None:
            self.num_heads = 1
        else:
            self.num_heads = num_heads
        self.head_size = hidden_dim // self.num_heads
        self.dropout = dropout
        self.bias = bias
        self.temperature = temperature
        self.use_se = use_se
        
        # Initialize positional embedding method
        if self.positional_embedding == "dyn":
            self.dynpos = DynamicPositionBias(
                dim=hidden_dim // 4,
                heads=num_heads, 
                depth=2
            )
        elif self.positional_embedding == "alibi":
            alibi_heads = num_heads // 2 + (num_heads % 2 == 1)
            self.alibi = AlibiPositionalBias(
                alibi_heads, 
                self.num_heads
            )
        elif self.positional_embedding == "xpos":
            self.xpos = XPOS(self.head_size)
            
        assert hidden_dim == self.head_size * self.num_heads, "hidden_dim must be divisible by num_heads"
        
        # Attention layers
        self.dropout_layer = nn.Dropout(dropout)
        self.weights = nn.Parameter(
            torch.empty(self.hidden_dim, 3 * self.hidden_dim)  # Q, K, V of equal sizes
        )
        self.out_w = nn.Parameter(
            torch.empty(self.hidden_dim, self.hidden_dim)
        )
        
        # Initialize biases if needed
        if self.bias:
            self.out_bias = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.in_bias = nn.Parameter(torch.zeros(1, 1, 3 * self.hidden_dim))
            
        # Initialize weights with Xavier/Glorot
        torch.nn.init.xavier_normal_(self.weights)
        torch.nn.init.xavier_normal_(self.out_w)
        
        # Scale parameter for attention
        if not use_se:
            self.gamma = nn.Parameter(torch.ones(self.num_heads).view(1, -1, 1, 1))

    def forward(self, x, adj, mask=None, same=True, return_attn_weights=False):
        """Forward pass for multi-head self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            adj: Adjacency matrix
            mask: Attention mask
            same: Whether Q and K are from the same sequence
            return_attn_weights: Whether to return attention weights
            
        Returns:
            Attention output and optionally attention weights
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project input to Q, K, V
        x = x @ self.weights
        if self.bias:
            x = x + self.in_bias
            
        # Split into Q, K, V and reshape for multi-head attention
        Q, K, V = x.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3).chunk(3, dim=3)
        
        # Apply positional encoding if needed
        if self.positional_embedding == "xpos":
            Q, K = self.xpos(Q), self.xpos(K, downscale=True)
        
        # Calculate attention scores
        norm = self.head_size**0.5
        attention = (Q @ K.transpose(2, 3) / self.temperature / norm)
        
        # Apply positional bias
        if self.positional_embedding == "dyn":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.dynpos(i, j).unsqueeze(0)
            attention = attention + attn_bias
        elif self.positional_embedding == "alibi":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.alibi(i, j).unsqueeze(0)
            attention = attention + attn_bias
        
        # Add adjacency information
        if not self.use_se:
            attention = attention + self.gamma * adj
        else:
            attention = attention + adj
        
        # Apply softmax
        attention = attention.softmax(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            attention = attention * mask.view(batch_size, 1, 1, -1)
            
        # Calculate output
        out = attention @ V
        out = out.permute(0, 2, 1, 3).flatten(2, 3)
        
        # Apply output projection
        out = out @ self.out_w.t()
        if self.bias:
            out = out + self.out_bias
            
        if return_attn_weights:
            return out, attention
        else:
            return out


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward network."""
    def __init__(
        self,
        hidden_dim: int,
        positional_embedding: str,
        num_heads: Optional[int] = None,
        dropout: float = 0.10,
        ffn_size: Optional[int] = None,
        activation: nn.Module = nn.GELU,
        temperature: float = 1.,
        use_se: bool = False,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = 1
        if ffn_size is None:
            ffn_size = hidden_dim * 4
            
        # Layer normalization and attention
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.mhsa = MultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            positional_embedding=positional_embedding,
            dropout=dropout,
            bias=True,
            temperature=temperature,
            use_se=use_se,
        )
        
        # Dropout for residual connections
        self.dropout_layer = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_size),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, adj, mask=None, return_attn_weights=False):
        """Forward pass for the transformer encoder layer.
        
        Args:
            x: Input tensor
            adj: Adjacency matrix
            mask: Attention mask
            return_attn_weights: Whether to return attention weights
            
        Returns:
            Transformer layer output and optionally attention weights
        """
        # Save input for residual connection
        x_in = x
        
        # Apply attention with layer norm
        if return_attn_weights:
            x, attn_w = self.mhsa(self.in_norm(x), adj=adj, mask=mask, return_attn_weights=True)
        else:
            x = self.mhsa(self.in_norm(x), adj=adj, mask=mask, return_attn_weights=False)
            
        # First residual connection
        x = self.dropout_layer(x) + x_in
        
        # FFN with second residual connection
        x = self.ffn(x) + x

        if return_attn_weights:
            return x, attn_w
        else:
            return x


class SE_Block(nn.Module):
    """Squeeze-and-Excitation block for feature recalibration."""
    def __init__(self, c, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply squeeze and excitation to input tensor.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Recalibrated feature maps
        """
        batch_size, channels, _, _ = x.shape
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class ResConv2dSimple(nn.Module):
    """Simple residual convolutional block."""
    def __init__(
        self, 
        in_c, 
        out_c,
        kernel_size=7,
        use_se=False,
    ):  
        super().__init__()
        
        # Create convolution block with optional SE
        if use_se:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding="same", bias=False),
                nn.BatchNorm2d(out_c),
                SE_Block(out_c),
                nn.GELU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding="same", bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
            )
        
        # Residual connection
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)

    def forward(self, x):
        """Apply residual convolution.
        
        Args:
            x: Input tensor
            
        Returns:
            Output after residual convolution
        """
        h = self.conv(x)
        x = self.res(x) + h
        return x


class AdjTransformerEncoder(nn.Module):
    """Transformer encoder with adjacency matrix processing."""
    def __init__(
        self,
        positional_embedding: str,
        dim: int = 192,
        head_size: int = 32,
        dropout: float = 0.10,
        dim_feedforward: int = 768,  # 192 * 4
        activation: nn.Module = nn.GELU,
        temperature: float = 1.,
        num_layers: int = 12,
        num_adj_convs: int = 3,
        ks: int = 3,
        use_se: bool = False,
    ):
        super().__init__()
        print(f"Using kernel size {ks}")
        
        # Calculate number of heads
        num_heads, rest = divmod(dim, head_size)
        assert rest == 0
        self.num_heads = num_heads
        
        # Create transformer layers
        self.layers = nn.Sequential(
            *[TransformerEncoderLayer(
                hidden_dim=dim,
                num_heads=num_heads,
                positional_embedding=positional_embedding,
                dropout=dropout,
                ffn_size=dim_feedforward,
                activation=activation,
                temperature=temperature,
                use_se=use_se,
            ) for i in range(num_layers)]
        )
        
        # Create convolutional layers for adjacency matrix processing
        self.conv_layers = nn.ModuleList()
        for i in range(num_adj_convs):
            self.conv_layers.append(
                ResConv2dSimple(
                    in_c=1 if i == 0 else num_heads,
                    out_c=num_heads,
                    kernel_size=ks,
                    use_se=use_se
                )
            )

    def forward(self, x, adj, mask=None):
        """Forward pass for the transformer encoder.
        
        Args:
            x: Input tensor
            adj: Adjacency matrix
            mask: Attention mask
            
        Returns:
            Encoded representation
        """
        # Process adjacency matrix
        adj = torch.log(adj + 1e-5)
        adj = adj.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
        
        # Apply transformer layers with processed adjacency
        for ind, mod in enumerate(self.layers):
            if ind < len(self.conv_layers):
                conv = self.conv_layers[ind]
                adj = conv(adj)
            x = mod(x, adj=adj, mask=mask)

        return x


class RNAdjNetBrk(nn.Module):
    """RNA structure prediction network with bracket annotations."""
    def __init__(
        self,  
        positional_embedding: str,
        adj_ks: int,
        not_slice: bool,
        brk_names: Optional[List[str]] = None,
        num_convs: Optional[int] = None,
        dim: int = 192, 
        depth: int = 12,
        head_size: int = 32,
        brk_symbols: int = 9,
        use_se: bool = False,
    ):
        super().__init__()
        self.slice_tokens = not not_slice 
        if not self.slice_tokens:
            print("Not removing unnecessary padding tokens. This can downgrade performance and slow the training")
        else:
            print("Removing unnecessary padding tokens")
            
        print(f"Using {positional_embedding} positional embedding")
        if num_convs is None:
            num_convs = depth
        print(f"Using {num_convs} conv layers")
        
        # Embeddings for nucleotides and special tokens
        self.emb = nn.Embedding(4 + 3, dim)  # 4 nucleotides + 3 tokens
        self.brk_names = brk_names
        print('Using', brk_names)
        
        # Main transformer encoder
        self.transformer = AdjTransformerEncoder(
            num_layers=depth,
            num_adj_convs=num_convs,
            dim=dim,
            head_size=head_size,
            positional_embedding=positional_embedding,
            ks=adj_ks,
            use_se=use_se,
        )
        
        # Output projection
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2)
        )
        
        # Structure embeddings
        self.struct_embeds = nn.ModuleDict()
        
        if self.brk_names is not None:
            for method in self.brk_names:
                emb = nn.Embedding(brk_symbols + 3, dim)
                self.struct_embeds[method] = emb
            self.struct_embeds_weights = nn.Parameter(torch.ones(len(brk_names)))
            
        # Embedding for sequence quality
        self.is_good_embed = nn.Embedding(2, dim)
            
    def forward(self, x0):
        """Forward pass for RNA structure prediction.
        
        Args:
            x0: Dictionary containing input data
            
        Returns:
            Prediction tensor
        """
        # Get mask and possibly slice
        mask = x0['forward_mask']
        if self.slice_tokens:
            Lmax = mask.sum(-1).max()
            mask = mask[:, :Lmax]
            
        # Process adjacency matrix
        adj = x0['adj'] 
        if self.slice_tokens:
            adj = adj[:, :Lmax, :Lmax]      
        
        # Get sequence embeddings
        if self.slice_tokens:
            e = self.emb(x0['seq_int'][:, :Lmax])
        else:
            e = self.emb(x0['seq_int'])
    
        # Combine embeddings
        x = e
        
        # Add quality embedding
        is_good = x0['is_good']
        e_is_good = self.is_good_embed(is_good)  # [batch_size, dim]
        e_is_good = e_is_good.unsqueeze(1)  # [batch_size, 1, dim]
        x = x + e_is_good
        
        # Add structure embeddings if available
        if self.brk_names is not None:
            for ind, method in enumerate(self.brk_names):
                st = x0[method]
                if self.slice_tokens:
                    st = st[:, :Lmax]
                st_embed = self.struct_embeds[method](st)
                x = x + st_embed * self.struct_embeds_weights[ind]
        
        # Apply transformer
        x = self.transformer(x, adj, mask=mask)
        
        # Project to output space
        x = self.proj_out(x)
   
        return x


class BPPFeatures:
    """Class for base pair probability features."""
    LMAX: ClassVar[int] = 206

    def __init__(self, index_path: str, mempath: str):
        self.index = self.read_index(index_path)
        self.storage = self.read_memmap(mempath, len(self.index))
        
    @classmethod
    def read_index(cls, index_path):
        """Read index file for sequences.
        
        Args:
            index_path: Path to index file
            
        Returns:
            Dictionary mapping sequence IDs to indices
        """
        with open(index_path) as inp:
            ids = [line.strip() for line in inp]
        index = {seqid: i for i, seqid in enumerate(ids)}
        return index
    
    @classmethod
    def read_memmap(cls, memmap_path, index_len):
        """Read memory-mapped file for base pair probabilities.
        
        Args:
            memmap_path: Path to memmap file
            index_len: Length of index
            
        Returns:
            Numpy memmap array
        """
        storage = np.memmap(
            memmap_path, 
            dtype=np.float32,
            mode='r', 
            offset=0,
            shape=(index_len, cls.LMAX, cls.LMAX),
            order='C'
        )
        return storage
    
    def __getitem__(self, seqid):
        """Get base pair probabilities for a sequence.
        
        Args:
            seqid: Sequence ID
            
        Returns:
            Base pair probability matrix
        """
        ind = self.index[seqid]
        return self.storage[ind]


class MISSING:
    """Sentinel class for missing values."""
    pass


def load_eterna(seq_id: str, maxL: int):
    """Load Eterna dataset base pair probabilities.
    
    Args:
        seq_id: Sequence ID
        maxL: Maximum sequence length
        
    Returns:
        Padded base pair probability matrix
    """
    path = BPP_ROOT_DIR / f"{seq_id}.npy"
    mat = np.load(path)
    dif = maxL - mat.shape[0]
    res = np.pad(mat, ((0, dif), (0, dif)))
    return res


class RNA_Dataset(Dataset):
    """Dataset for RNA sequences and structures."""
    def __init__(
        self, 
        df,
        seq_structs: Dict[str, Dict[str, str]],
        split_type: str,
        Lmax: int,
        use_shift: bool,
        use_reverse: bool,
        train_threshold: Optional[Union[str, MISSING]] = MISSING,
        mode: str = 'train', 
        seed: int = 2023, 
        fold: int = 0, 
        nfolds: int = 4
    ):
        if mode == "train" and train_threshold is MISSING:
            raise Exception("Train threshold should be specified for train mode")
            
        # Mapping dictionaries
        self.seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, "START": 4, "END": 5, "EMPTY": 6}
        self.brk_map = {
            "(": 0, ")": 1, "[": 2, "]": 3, "{": 4, "}": 5,
            "<": 6, ">": 7, ".": 8, "START": 9, "END": 10, "EMPTY": 11
        }
        
        assert mode in ('train', 'eval')
        self.Lmax = Lmax
        self.seq_structs = seq_structs
        df['L'] = df.sequence.apply(len)

        # Split data based on split_type
        if split_type == "kfold":
            df_2A3 = df.loc[df.experiment_type == '2A3_MaP']
            df_DMS = df.loc[df.experiment_type == 'DMS_MaP']
            split_indices = list(KFold(n_splits=nfolds, random_state=seed, shuffle=True).split(df_2A3))[fold]
            indices = split_indices[0 if mode == 'train' else 1]
            df_2A3 = df_2A3.iloc[indices].reset_index(drop=True)
            df_DMS = df_DMS.iloc[indices].reset_index(drop=True)
        elif split_type == "length":
            if mode == "eval":
                df = df[df['L'] >= 206]
            else:
                df = df[df['L'] < 206]
            print(mode, df.shape)
                
            df_2A3 = df.loc[df.experiment_type == '2A3_MaP']
            df_DMS = df.loc[df.experiment_type == 'DMS_MaP']
        else:
            raise Exception(f"Unknown split type: {split_type}")

        print("2A3 shape before filter", df_2A3.shape, "threshold", train_threshold, "split", split_type, "mode", mode)
        
        # Filter data based on mode and threshold
        if mode == "eval":
            print("Keeping only clean data for validation")
            mask = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
            df_2A3 = df_2A3.loc[mask].reset_index(drop=True)
            df_DMS = df_DMS.loc[mask].reset_index(drop=True)
        elif mode == "train":
            if train_threshold is not None:
                print(f"Using threshold {train_threshold}")
                if train_threshold == "clean":
                    mask = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
                else:
                    try:
                        threshold_val = float(train_threshold)
                        mask = np.logical_and(
                            df_2A3['signal_to_noise'].values >= threshold_val,  
                            df_DMS['signal_to_noise'].values >= threshold_val
                        )
                    except ValueError:
                        raise Exception("Threshold must be None, float or 'clean'")
                df_2A3 = df_2A3.loc[mask].reset_index(drop=True)
                df_DMS = df_DMS.loc[mask].reset_index(drop=True)
            else:
                print(f"Using no threshold")
        
        print("2A3 shape after filter", df_2A3.shape, "threshold", train_threshold, "split", split_type, "mode", mode)
        
        # Extract data from dataframes
        self.sid = df_2A3['sequence_id'].values
        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values
        
        # Extract reactivity and error values
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if 'reactivity_error_0' in c]].values
        
        # Signal-to-noise metrics
        self.is_good = ((df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)) * 1
        self.sn_2A3 = df_2A3['SN_filter'].values 
        self.sn_DMS = df_DMS['SN_filter'].values
        
        # Calculate sample weights based on signal-to-noise ratio
        sn = (df_2A3['signal_to_noise'].values + df_DMS['signal_to_noise'].values) / 2
        sn = torch.from_numpy(sn)
        self.weights = 0.5 * torch.clamp_min(torch.log(sn + 1.01), 0.01)
        
        # Set dataset parameters
        self.mode = mode
        self.use_shift = use_shift
        if self.use_shift:
            print("Use shifting")
        self.use_reverse = use_reverse
        if self.use_reverse:
            print("Use reverse")
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.seq)  
    
    def _process_seq(self, rawseq, shift):
        """Process raw sequence into model input format with padding.
        
        Args:
            rawseq: Raw RNA sequence
            shift: Amount of shift for data augmentation
            
        Returns:
            Processed sequence tensor and positions
        """
        seq = [self.seq_map['EMPTY'] for _ in range(shift)]
        seq.append(self.seq_map['START'])
        start_loc = len(seq) - 1
        seq.extend(self.seq_map[s] for s in rawseq)
        seq.append(self.seq_map['END'])
        end_loc = len(seq) - 1
        
        # Pad to maximum length
        seq.extend([self.seq_map['EMPTY']] * (self.Lmax + 2 - len(seq)))
            
        seq = torch.tensor(seq, dtype=torch.long)
        return seq, start_loc, end_loc
    
    def _process_brk(self, rawbrk, shift):
        """Process raw brackets notation into model input format.
        
        Args:
            rawbrk: Raw bracket notation
            shift: Amount of shift for data augmentation
            
        Returns:
            Processed bracket tensor
        """
        brk = [self.brk_map['EMPTY'] for _ in range(shift)]
        brk.append(self.brk_map['START'])
        brk.extend(self.brk_map[b] for b in rawbrk)
        brk.append(self.brk_map['END'])
        
        # Pad to maximum length
        brk.extend([self.brk_map['EMPTY']] * (self.Lmax + 2 - len(brk)))
         
        brk = torch.tensor(brk, dtype=torch.long)
        return brk
    
    def get_shift(self, seqL):
        """Determine random shift amount for data augmentation.
        
        Args:
            seqL: Sequence length
            
        Returns:
            Shift amount (0 for evaluation mode)
        """
        if not self.use_shift or self.mode == "eval":
            return 0
        
        dif = self.Lmax - seqL 
        shift = torch.randint(low=0, high=dif+1, size=(1,)).item()  # high is not included
        return shift
    
    def get_to_rev(self):
        """Determine whether to reverse the sequence for data augmentation.
        
        Returns:
            Boolean indicating whether to reverse
        """
        if not self.use_reverse or self.mode == "eval":
            return False
        
        return torch.rand(1).item() > 0.5
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Model inputs and targets
        """
        seq = self.seq[idx]
        real_seq_L = len(seq)
        
        # Get augmentation parameters
        shift = self.get_shift(real_seq_L)
        
        # Calculate padding borders
        lbord = 1 + shift
        rbord = self.Lmax + 1 - real_seq_L - shift
        
        # Process sequence
        seq_int, start_loc, end_loc = self._process_seq(seq, shift)
        
        # Create masks
        mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)
        mask[start_loc+1:end_loc] = True  # not including START and END
      
        forward_mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)  # START, seq, END
        forward_mask[start_loc:end_loc+1] = True  # including START and END
        
        # Process reactivity data
        react = np.stack([
            self.react_2A3[idx][:real_seq_L],
            self.react_DMS[idx][:real_seq_L]
        ], -1)
        react = np.pad(react, ((lbord, rbord), (0, 0)), constant_values=np.nan)
        react = torch.from_numpy(react)
     
        # Create input dictionary
        X = {
            'seq_int': seq_int,
            'mask': mask, 
            "forward_mask": forward_mask,
            'is_good': self.is_good[idx]
        }
        
        # Process bracket structures
        sid = self.sid[idx]
        for method, structs in self.seq_structs.items():
            brk = structs[sid]
            X[method] = self._process_brk(brk, shift)
        
        # Process adjacency matrix
        adj = load_eterna(sid, self.Lmax)[:real_seq_L, :real_seq_L]
        adj = np.pad(adj, ((lbord, rbord), (lbord, rbord)), constant_values=0)
        adj = torch.from_numpy(adj).float()
        X['adj'] = adj
        
        # Create target dictionary
        y = {
            'react': react.float(), 
            'mask': mask
        }
        
        # Apply sequence reversal if needed
        to_rev = self.get_to_rev()
        if to_rev:
            X_rev = {}
            for key, value in X.items():
                if key == "is_good":
                    X_rev[key] = value
                elif key in ("seq_int", "mask", "forward_mask"):
                    X_rev[key] = value.flip(dims=[0])
                elif key == "adj":
                    X_rev[key] = value.flip(dims=[0, 1])
                elif key in self.seq_structs:
                    X_rev[key] = value.flip(dims=[0])
                else:
                    raise Exception(f"No reverse process for key {key}")
            X = X_rev
                
            y_rev = {}
            for key, value in y.items():
                if key == "mask":
                    y_rev[key] = value.flip(dims=[0])
                elif key == "react":
                    y_rev[key] = value.flip(dims=[0])
                else:
                    raise Exception(f"No reverse process for key {key}")
            y = y_rev
        
        return X, y


def loss(pred, target):
    """Calculate L1 loss for RNA reactivity prediction.
    
    Args:
        pred: Model predictions
        target: Ground truth targets
        
    Returns:
        L1 loss value
    """
    # Extract masked values
    p = pred[target['mask'][:, :pred.shape[1]]]
    y = target['react'][target['mask']].clip(0, 1)
    
    # Calculate L1 loss
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    
    return loss


class MAE(Metric):
    """Mean Absolute Error metric for RNA reactivity prediction."""
    
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        """Reset metric accumulation."""
        self.x, self.y = [], []
        
    def accumulate(self, learn):
        """Accumulate predictions and targets.
        
        Args:
            learn: Learner object with predictions and targets
        """
        x = learn.pred[learn.y['mask'][:, :learn.pred.shape[1]]]
        y = learn.y['react'][learn.y['mask']].clip(0, 1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        """Calculate mean absolute error.
        
        Returns:
            MAE value
        """
        x, y = torch.cat(self.x, 0), torch.cat(self.y, 0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss
    

class MAE_2A3(Metric):
    """Mean Absolute Error metric for 2A3 reactivity."""
    
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        """Reset metric accumulation."""
        self.x, self.y = [], []
        
    def accumulate(self, learn):
        """Accumulate predictions and targets for 2A3.
        
        Args:
            learn: Learner object with predictions and targets
        """
        x = learn.pred[:, :, 0][learn.y['mask'][:, :learn.pred.shape[1]]]
        y = learn.y['react'][:, :, 0][learn.y['mask']].clip(0, 1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        """Calculate mean absolute error for 2A3.
        
        Returns:
            MAE value for 2A3
        """
        x, y = torch.cat(self.x, 0), torch.cat(self.y, 0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss
    

class MAE_DMS(Metric):
    """Mean Absolute Error metric for DMS reactivity."""
    
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        """Reset metric accumulation."""
        self.x, self.y = [], []
        
    def accumulate(self, learn):
        """Accumulate predictions and targets for DMS.
        
        Args:
            learn: Learner object with predictions and targets
        """
        x = learn.pred[:, :, 1][learn.y['mask'][:, :learn.pred.shape[1]]]
        y = learn.y['react'][:, :, 1][learn.y['mask']].clip(0, 1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        """Calculate mean absolute error for DMS.
        
        Returns:
            MAE value for DMS
        """
        x, y = torch.cat(self.x, 0), torch.cat(self.y, 0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss
    

def seed_everything(seed):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def val_to(x, device="cuda"):
    """Move values to specified device.
    
    Args:
        x: Value or list of values
        device: Target device
        
    Returns:
        Value(s) moved to device
    """
    if isinstance(x, list):
        return [val_to(z, device) for z in x]
    return x.to(device)


def dict_to(x, device='cuda'):
    """Move dictionary values to specified device.
    
    Args:
        x: Dictionary with tensor values
        device: Target device
        
    Returns:
        Dictionary with values moved to device
    """
    return {k: val_to(x[k], device) for k in x}


def to_device(x, device='cuda'):
    """Move tuple of dictionaries to device.
    
    Args:
        x: Tuple of dictionaries
        device: Target device
        
    Returns:
        Tuple with dictionaries moved to device
    """
    return tuple(dict_to(e, device) for e in x)


class DeviceDataLoader:
    """DataLoader that automatically moves batches to the specified device."""
    
    def __init__(self, dataloader, device='cuda'):
        """Initialize DeviceDataLoader.
        
        Args:
            dataloader: PyTorch DataLoader
            device: Target device
        """
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        """Return the number of batches."""
        return len(self.dataloader)
    
    def __iter__(self):
        """Iterate through batches, moving each to device."""
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)


# Main script for RNA structure prediction

import argparse

def main():
    """Main function to parse arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(description="RNA structure prediction training script")
    
    # Data paths
    parser.add_argument("--bpp_path", required=True, type=str,
                        help="Path to base pair probability files")
    parser.add_argument("--train_path", required=True, type=str,
                        help="Path to training data file (parquet)")
    parser.add_argument("--brackets", required=False, default=[], type=str, nargs='+',
                        help="Paths to bracket structure JSON files")
    parser.add_argument("--out_path", required=True, type=str,
                        help="Path to save output models")
    
    # Training configuration
    parser.add_argument("--num_workers", default=32, type=int,
                        help="Number of data loading workers")
    parser.add_argument("--nfolds", default=4, type=int,
                        help="Number of cross-validation folds")
    parser.add_argument("--fold", default=0, type=int,
                        help="Current fold to train")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Training batch size")
    parser.add_argument("--device", default=0, type=int,
                        help="CUDA device index")
    parser.add_argument("--seed", default=2023, type=int,
                        help="Random seed for reproducibility")
    parser.add_argument("--epoch", default=200, type=int,
                        help="Number of training epochs")
    
    # Optimizer settings
    parser.add_argument("--lr_max", default=2.5e-3, type=float,
                        help="Maximum learning rate")
    parser.add_argument("--wd", default=0.05, type=float,
                        help="Weight decay")
    parser.add_argument("--pct_start", default=0.05, type=float,
                        help="Percentage of training to increase learning rate")
    parser.add_argument("--gradclip", default=1.0, type=float,
                        help="Gradient clipping value")
    
    # Model architecture
    parser.add_argument("--num_attn_layers", default=12, type=int,
                        help="Number of attention layers")
    parser.add_argument("--num_conv_layers", default=12, type=int,
                        help="Number of convolutional layers")
    parser.add_argument("--adj_ks", required=True, type=int,
                        help="Kernel size for adjacency convolutions")
    
    # Dataset configuration
    parser.add_argument("--split", default="kfold", type=str, choices=["kfold", "length"],
                        help="Data split type")
    parser.add_argument("--batch_cnt", default=1791, type=int,
                        help="Number of batches per epoch")
    parser.add_argument("--Lmax", default=206, type=int,
                        help="Maximum sequence length")
    parser.add_argument("--use_shift", action="store_true",
                        help="Use shift augmentation")
    parser.add_argument("--use_reverse", action="store_true",
                        help="Use reverse augmentation")
    
    # SGD fine-tuning
    parser.add_argument("--sgd_lr", default=5e-5, type=float,
                        help="SGD learning rate")
    parser.add_argument("--sgd_epochs", default=25, type=int,
                        help="Number of SGD epochs")
    parser.add_argument("--sgd_batch_cnt", default=500, type=int,
                        help="Number of SGD batches per epoch")
    parser.add_argument("--sgd_wd", default=0.05, type=float,
                        help="SGD weight decay")
    
    # Model options
    parser.add_argument("--pos_embedding", choices=['xpos', 'dyn', 'alibi'], required=True,
                        help="Positional embedding type")
    parser.add_argument("--use_se", action="store_true",
                        help="Use Squeeze-and-Excitation blocks")
    parser.add_argument("--train_threshold", default=None,
                        help="Signal-to-noise threshold for training data")
    parser.add_argument("--pretrained_model", default=None,
                        help="Path to pretrained model weights")
    parser.add_argument("--not_slice", action="store_true",
                        help="Do not slice unnecessary padding tokens")
    
    args = parser.parse_args()
    print(args)
    
    # Set global paths
    global BPP_ROOT_DIR
    BPP_ROOT_DIR = Path(args.bpp_path)
    
    # Load dataset
    import pandas as pd
    df = pd.read_parquet(args.train_path)
    
    # Load bracket structures
    ribo_dt = {}
    for br_path in args.brackets:
        n = Path(br_path).name.replace(".json", "")
        with open(br_path) as inp:
            dt = json.load(inp)
            ribo_dt[n] = dt    
    
    # Create output directory
    OUT = args.out_path
    os.makedirs(OUT, exist_ok=True)
    fname = 'model'
    
    # Device setup
    device = torch.device(f"cuda:{args.device}")
    
    # Set up fastai callbacks
    smclbk = SaveModelCallback(
        monitor='valid_loss',
        fname='model', 
        with_opt=True
    )
    
    p = Path(f'{OUT}/fst_model/pos_aug')
    p.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(fname=str(p / "loss.csv"))
    
    # Set random seed for reproducibility
    SEED = args.seed
    seed_everything(SEED)
    
    # Train the model for specified fold
    fold = args.fold
    print(f"Training fold {fold}")
    
    # Create training dataset
    print("Loading training data")
    ds_train = RNA_Dataset(
        df, 
        use_shift=args.use_shift,
        use_reverse=args.use_reverse,
        Lmax=args.Lmax, 
        train_threshold=args.train_threshold,
        seq_structs=ribo_dt, 
        split_type=args.split,
        mode='train', 
        fold=fold, 
        nfolds=args.nfolds
    )
    
    # Create training sampler with weighted sampling
    sampler_train = WeightedRandomSampler(
        weights=ds_train.weights, 
        num_samples=args.batch_size * args.batch_cnt
    )
    
    # Create training dataloader
    dl_train = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_train, 
            batch_size=args.batch_size,
            sampler=sampler_train, 
            num_workers=args.num_workers,
            persistent_workers=True
        ), 
        device
    )
    
    # Create validation dataset
    print("Loading validation data")
    ds_val = RNA_Dataset(
        df, 
        use_shift=args.use_shift,
        use_reverse=args.use_reverse,
        Lmax=args.Lmax, 
        seq_structs=ribo_dt, 
        split_type=args.split,
        mode='eval',
        fold=fold,
        nfolds=args.nfolds
    )
    
    # Create validation dataloader
    dl_val = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_val, 
            batch_size=args.batch_size * 4,
            num_workers=args.num_workers,
            persistent_workers=True
        ), 
        device
    )
    
    # Clean up memory
    gc.collect()
    
    # Create fastai DataLoaders
    data = DataLoaders(dl_train, dl_val)
    
    # Initialize model
    model = RNAdjNetBrk(
        positional_embedding=args.pos_embedding,
        brk_names=list(ribo_dt.keys()),
        depth=args.num_attn_layers, 
        num_convs=args.num_conv_layers,
        adj_ks=args.adj_ks,
        not_slice=args.not_slice,
        use_se=args.use_se
    )
    print(model)
    
    # Load pretrained weights if specified
    if args.pretrained_model is not None:
        print(f"Loading pretrained model from {args.pretrained_model}")
        model.load_state_dict(
            torch.load(
                args.pretrained_model,
                map_location="cpu"
            )['model']
        )
    
    # Move model to device
    model = model.to(device)
    
    # Create fastai learner
    learn = Learner(
        data, 
        model, 
        loss_func=loss,
        model_dir=p,
        cbs=[
            GradientClip(args.gradclip), 
            logger, 
            smclbk
        ],
        metrics=[MAE(), MAE_DMS(), MAE_2A3()]
    ).to_fp16()  # fp16 for modern hardware speedup
    
    # Train with one-cycle policy
    print("Starting main training cycle")
    learn.fit_one_cycle(
        args.epoch,
        lr_max=args.lr_max,
        wd=args.wd,
        pct_start=args.pct_start
    )
    
    # Save model weights
    torch.save(
        learn.model.state_dict(),
        os.path.join(OUT, f'{fname}_{fold}.pth')
    )
    
    # Clean up memory
    gc.collect()
    
    # Fine-tune with SGD
    print("Starting SGD fine-tuning")
    
    # Create new sampler for SGD
    sampler_train = WeightedRandomSampler(
        weights=ds_train.weights, 
        num_samples=args.batch_size * args.sgd_batch_cnt
    )
    
    # Create new dataloader for SGD
    dl_train = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_train, 
            batch_size=args.batch_size,
            sampler=sampler_train, 
            num_workers=args.num_workers,
            persistent_workers=True
        ), 
        device
    )
    
    # Create new validation dataloader
    dl_val = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_val, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=True
        ), 
        device
    )
    
    # Create fastai DataLoaders for SGD
    data = DataLoaders(dl_train, dl_val)
    
    # Set up SGD output directory
    p = Path(f'{OUT}/fst_model/pos_aug_sgd')
    p.mkdir(parents=True, exist_ok=True)
    
    # Set up SGD callbacks
    smclbk = SaveModelCallback(
        monitor='valid_loss',
        fname='model', 
        with_opt=True,
    )
    logger = CSVLogger(fname=str(p / "loss.csv"))
    
    # Create SGD learner
    learn = Learner(
        data,
        model,
        model_dir=p,
        lr=args.sgd_lr,
        opt_func=partial(OptimWrapper, opt=torch.optim.SGD),
        loss_func=loss,
        cbs=[GradientClip(args.gradclip), smclbk, logger],
        metrics=[MAE(), MAE_DMS(), MAE_2A3()]
    ).to_fp16()
    
    # Train with SGD
    print("Training with SGD")
    learn.fit(
        args.sgd_epochs, 
        wd=args.sgd_wd
    )
    
    # Save final model
    torch.save(
        learn.model.state_dict(),
        os.path.join(OUT, f'{fname}_{fold}_sgd.pth')
    )
    
    print(f"Training completed for fold {fold}")


if __name__ == "__main__":
    main()