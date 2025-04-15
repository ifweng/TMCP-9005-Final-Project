import os
import gc
import math
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple, Optional, Union, Any
import json
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

from xpos_relative_position import XPOS


def parse_args():
    """Parse command line arguments for RNA structure prediction."""
    parser = argparse.ArgumentParser(description="RNA Structure Prediction")
    parser.add_argument("--bpp_path",
                        required=True,
                        type=str,
                        help="Path to base pair probability files")
    parser.add_argument("--test_path", 
                        required=True,
                        type=str,
                        help="Path to test data CSV")
    parser.add_argument("--model_path", 
                        required=True,
                        type=str,
                        help="Path to pretrained model weights")
    parser.add_argument("--out_path", 
                        required=True,
                        type=str,
                        help="Output directory for predictions")
    parser.add_argument("--brackets", 
                        required=False,
                        default=[],
                        type=str, 
                        nargs='+',
                        help="Paths to bracket structure JSON files")
    parser.add_argument("--fold",  #solely for output naming
                        default=0,
                        type=int,
                        help="Fold number (for output naming)")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="Batch size for inference")
    parser.add_argument("--num_workers",
                        default=64,
                        type=int,
                        help="Number of dataloader workers")
    parser.add_argument("--device",
                        default=0,
                        type=int,
                        help="CUDA device index")
    parser.add_argument("--pos_embedding",
                        choices=['xpos', 'dyn', 'alibi'],
                        required=True,
                        help="Type of positional embedding")
    parser.add_argument("--num_attn_layers",
                        default=12,
                        type=int,
                        help="Number of attention layers")
    parser.add_argument("--num_conv_layers",
                        default=12,
                        type=int,
                        help="Number of convolution layers")
    parser.add_argument("--adj_ks",
                        required=True, 
                        type=int,
                        help="Kernel size for adjacency convolutions")
    parser.add_argument("--not_slice",
                        action="store_true",
                        help="Don't remove padding tokens")
    parser.add_argument("--use_se",
                        action="store_true",
                        help="Use Squeeze-and-Excitation blocks")
    
    return parser.parse_args()


# Utility functions
def exists(val: Any) -> bool:
    """Check if a value exists (is not None)."""
    return val is not None


def pad_at_dim(t: torch.Tensor, pad: Tuple[int, int], dim: int = -1, value: float = 0.) -> torch.Tensor:
    """Pad a tensor at a specific dimension."""
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def val_to(x: Any, device: torch.device) -> Any:
    """Move a value or list of values to the specified device."""
    if isinstance(x, list):
        return [val_to(z, device) for z in x]
    return x.to(device)


def dict_to(x: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all values in a dictionary to the specified device."""
    return {k: val_to(x[k], device) for k in x}


def to_device(x: Tuple[Dict[str, Any], ...], device: torch.device) -> Tuple[Dict[str, Any], ...]:
    """Move a tuple of dictionaries to the specified device."""
    return tuple(dict_to(e, device) for e in x)


# Position Bias Models
class DynamicPositionBias(nn.Module):
    """Dynamic position bias for transformer attention."""
    def __init__(self, dim: int, *, heads: int, depth: int, log_distance: bool = False, norm: bool = False):
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
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, i: int, j: int) -> torch.Tensor:
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device=device)
        context_arange = torch.arange(n, device=device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device=device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias


class AlibiPositionalBias(nn.Module):
    """ALiBi positional bias for transformer attention."""
    def __init__(self, heads: int, total_heads: int, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)
    
    def get_bias(self, i: int, j: int, device: torch.device) -> torch.Tensor:
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads: int) -> List[float]:
        def get_slopes_power_of_2(n: int) -> List[float]:
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    @property
    def device(self) -> torch.device:
        return next(self.buffers()).device

    def forward(self, i: int, j: int) -> torch.Tensor:
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer('bias', bias, persistent=False)

        return self.bias


# Neural Network Components
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with support for different positional embeddings."""
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
        
        self.dropout_layer = nn.Dropout(dropout)
        self.weights = nn.Parameter(
            torch.empty(self.hidden_dim, 3 * self.hidden_dim)  # Q, K, V of equal sizes in given order
        )
        self.out_w = nn.Parameter(
            torch.empty(self.hidden_dim, self.hidden_dim)
        )
        if self.bias:
            self.out_bias = nn.Parameter(
                torch.empty(1, 1, self.hidden_dim)
            )
            torch.nn.init.constant_(self.out_bias, 0.)
            self.in_bias = nn.Parameter(
                torch.empty(1, 1, 3 * self.hidden_dim)
            )
            torch.nn.init.constant_(self.in_bias, 0.)
        torch.nn.init.xavier_normal_(self.weights)
        torch.nn.init.xavier_normal_(self.out_w)
        if not use_se:
            self.gamma = nn.Parameter(torch.ones(self.num_heads).view(1, -1, 1, 1))

    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        same: bool = True, 
        return_attn_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, l, h = x.shape
        x = x @ self.weights 
        if self.bias:
            x = x + self.in_bias  # b, l, 3*hidden
        Q, K, V = x.view(b, l, self.num_heads, -1).permute(0, 2, 1, 3).chunk(3, dim=3)  # b, a, l, head
        
        if self.positional_embedding == "xpos":
            Q, K = self.xpos(Q), self.xpos(K, downscale=True)
        
        norm = self.head_size**0.5
        attention = (Q @ K.transpose(2, 3) / self.temperature / norm)
        
        if self.positional_embedding == "dyn":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.dynpos(i, j).unsqueeze(0)
            attention = attention + attn_bias
        elif self.positional_embedding == "alibi":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.alibi(i, j).unsqueeze(0)
            attention = attention + attn_bias
        
        if not self.use_se:
            attention = attention + self.gamma * adj
        else:
            attention = attention + adj
        
        attention = attention.softmax(dim=-1)  # b, a, l, l
        
        if mask is not None:
            attention = attention * mask.view(b, 1, 1, -1) 
            
        out = attention @ V  # b, a, l, head
        out = out.permute(0, 2, 1, 3).flatten(2, 3)  # b, a, l, head -> b, l, (a, head) -> b, l, hidden
        
        out = out @ self.out_w
        if self.bias:
            out = out + self.out_bias
            
        if return_attn_weights:
            return out, attention
        else:
            return out


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head self-attention and feed-forward network."""
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
        self.dropout_layer = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_size),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        return_attn_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_in = x
        if return_attn_weights:
            x, attn_w = self.mhsa(self.in_norm(x), adj=adj, mask=mask, return_attn_weights=True)
        else:
            x = self.mhsa(self.in_norm(x), adj=adj, mask=mask, return_attn_weights=False)
        x = self.dropout_layer(x) + x_in
        x = self.ffn(x) + x

        if return_attn_weights:
            return x, attn_w
        else:
            return x


class SE_Block(nn.Module):
    """Squeeze and Excitation block for channel attention.
    
    Credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    """
    def __init__(self, c: int, r: int = 1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class ResConv2dSimple(nn.Module):
    """Residual 2D convolutional layer with optional SE blocks."""
    def __init__(
        self, 
        in_c: int, 
        out_c: int,
        kernel_size: int = 7,
        use_se: bool = False,
    ):  
        super().__init__()
        if use_se:
            self.conv = nn.Sequential(
                # b c w h
                nn.Conv2d(
                    in_c,
                    out_c, 
                    kernel_size=kernel_size, 
                    padding="same", 
                    bias=False
                ),
                # b w h c
                nn.BatchNorm2d(out_c),
                SE_Block(out_c),
                nn.GELU(),
                # b c e 
            )
        else:
            self.conv = nn.Sequential(
                # b c w h
                nn.Conv2d(
                    in_c,
                    out_c, 
                    kernel_size=kernel_size, 
                    padding="same", 
                    bias=False
                ),
                # b w h c
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                # b c e 
            )

        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(
                    in_c,
                    out_c, 
                    kernel_size=1, 
                    bias=False
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        dim_feedforward: int = 192 * 4,
        activation: nn.Module = nn.GELU,
        temperature: float = 1.,
        num_layers: int = 12,
        num_adj_convs: int = 3,
        ks: int = 3,
        use_se: bool = False,
    ):
        super().__init__()
        print(f"Using kernel size {ks}")
        num_heads, rest = divmod(dim, head_size)
        assert rest == 0
        self.num_heads = num_heads
        
        self.layers = nn.Sequential(
            *[TransformerEncoderLayer(
                hidden_dim=dim,
                num_heads=num_heads,
                positional_embedding=positional_embedding,
                dropout=dropout,
                ffn_size=dim_feedforward,
                activation=activation,
                temperature=temperature,
                use_se=use_se
            ) for i in range(num_layers)]
        )
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
            
    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # adj B S S 
        adj = torch.log(adj + 1e-5)
        adj = adj.unsqueeze(1)  # B 1 S S 
        
        for ind, mod in enumerate(self.layers):
            if ind < len(self.conv_layers):
                conv = self.conv_layers[ind]
                adj = conv(adj)
            x = mod(x, adj=adj, mask=mask)
            
        return x


class RNAdjNetBrk(nn.Module):
    """RNA adjacency network with bracket structures."""
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
        
        self.emb = nn.Embedding(4 + 3, dim)  # 4 nucleotides + 3 tokens
        self.brk_names = brk_names
        print('Using', brk_names)
        
        self.transformer = AdjTransformerEncoder(
            num_layers=depth,
            num_adj_convs=num_convs,
            dim=dim,
            head_size=head_size,
            positional_embedding=positional_embedding,
            ks=adj_ks,
            use_se=use_se,
        )
        
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2)
        )
        
        self.struct_embeds = nn.ModuleDict()
        
        if self.brk_names is not None:
            for method in self.brk_names:
                emb = nn.Embedding(brk_symbols + 3, dim)
                self.struct_embeds[method] = emb
            self.struct_embeds_weights = torch.nn.Parameter(torch.ones(len(brk_names)))
            
        self.is_good_embed = nn.Embedding(2, dim)
            
    def forward(self, x0: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = x0['forward_mask']
        if self.slice_tokens:
            Lmax = mask.sum(-1).max()
            mask = mask[:, :Lmax]
            
        adj = x0['adj'] 
        if self.slice_tokens:
            adj = adj[:, :Lmax, :Lmax]      
        
        if self.slice_tokens:
            e = self.emb(x0['seq_int'][:, :Lmax])
        else:
            e = self.emb(x0['seq_int'])
    
        x = e
        is_good = x0['is_good']
        e_is_good = self.is_good_embed(is_good)  # B E
        e_is_good = e_is_good.unsqueeze(1)  # B 1 E
        x = x + e_is_good
        
        if self.brk_names is not None:
            for ind, method in enumerate(self.brk_names):
                st = x0[method]
                if self.slice_tokens:
                    st = st[:, :Lmax]
                st_embed = self.struct_embeds[method](st)
                x = x + st_embed * self.struct_embeds_weights[ind]
                
        x = self.transformer(x, adj, mask=mask)
        
        x = self.proj_out(x)
   
        return x


class RNA_Dataset_Test(Dataset):
    """Dataset for RNA sequences with structure data for testing."""
    def __init__(self, df: pd.DataFrame, seq_structs: Dict[str, Dict[str, str]], mask_only: bool = False):
        df['L'] = df.sequence.apply(len)
        self.Lmax = df['L'].max()
        self.df = df
        self.mask_only = mask_only
        self.seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, "START": 4, "END": 5, "EMPTY": 6}
        self.brk_map = {
            "(": 0, ")": 1, "[": 2, "]": 3, "{": 4, "}": 5,
            "<": 6, ">": 7, ".": 8, "START": 9, "END": 10, "EMPTY": 11
        }
        self.seq_structs = seq_structs
        self.sid = df.sequence_id
        
    def __len__(self) -> int:
        return len(self.df)
    
    def _process_seq(self, rawseq: str) -> Tuple[torch.Tensor, int, int]:
        """Process a raw RNA sequence into token indices."""
        seq = []
        seq.append(self.seq_map['START'])
        start_loc = len(seq) - 1
        seq.extend(self.seq_map[s] for s in rawseq)
        seq.append(self.seq_map['END'])
        end_loc = len(seq) - 1
        for i in range(len(seq), self.Lmax + 2):
            seq.append(self.seq_map['EMPTY'])
            
        seq_array = np.array(seq)
        seq_tensor = torch.from_numpy(seq_array)
        
        return seq_tensor, start_loc, end_loc
    
    def _process_brk(self, rawbrk: str) -> torch.Tensor:
        """Process a raw bracket structure into token indices."""
        brk = [self.brk_map['START']]
        brk.extend(self.brk_map[b] for b in rawbrk)
        brk.append(self.brk_map['END'])
        brk_array = np.array(brk)
        brk_padded = np.pad(brk_array, (0, self.Lmax - len(brk) + 2), constant_values=self.brk_map['EMPTY'])
        brk_tensor = torch.from_numpy(brk_padded)
        return brk_tensor
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
        id_min, id_max, seq = self.df.loc[idx, ['id_min', 'id_max', 'sequence']]
        L = len(seq)
        
        if self.mask_only: 
            # Create and return only the mask for length estimation
            mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)
            mask[1:L+1] = True  # Not including START and END
            return {'mask': mask}, {}
        
        # Create sequence IDs array
        ids = np.arange(id_min, id_max + 1)
        ids = np.pad(ids, (1, self.Lmax + 1 - L), constant_values=-1)
        
        # Process sequence
        seq_int, start_loc, end_loc = self._process_seq(seq)
        
        # Create masks
        mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)
        mask[start_loc + 1:end_loc] = True  # Not including START and END
      
        forward_mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)  # START, seq, END
        forward_mask[start_loc:end_loc + 1] = True  # Including START and END
        
        # Build input dictionary
        X = {
            'seq_int': seq_int, 
            'mask': mask, 
            "is_good": 1,  
            "forward_mask": forward_mask
        }
        
        # Add structure information
        sid = self.sid[idx]
        for method, structs in self.seq_structs.items():
            try:
                brk = structs[sid]
            except KeyError:
                print(f"No struct {method} for {sid}")
                brk = self.seq_structs['eterna'][sid]
            X[method] = self._process_brk(brk)

        # Add adjacency matrix
        adj = load_bpp(sid, self.Lmax)
        adj = np.pad(adj, ((1, 1), (1, 1)), constant_values=0)
        X['adj'] = torch.from_numpy(adj).float()
        
        return X, {'ids': ids}


class DeviceDataLoader:
    """DataLoader wrapper that moves batches to a specified device."""
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self) -> int:
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)


def load_bpp(seq_id: str, maxL: int) -> np.ndarray:
    """Load base pair probability matrix for a sequence."""
    path = BPP_ROOT_DIR / f"{seq_id}.npy"
    mat = np.load(path)
    dif = maxL - mat.shape[0]
    res = np.pad(mat, ((0, dif), (0, dif)))
    return res


def load_bracket_structures(bracket_paths: List[str]) -> Dict[str, Dict[str, str]]:
    """Load bracket structure files from paths."""
    bracket_data = {}
    for br_path in bracket_paths:
        name = Path(br_path).name.replace(".json", "")
        with open(br_path) as inp:
            data = json.load(inp)
            bracket_data[name] = data
    
    return bracket_data


def run_inference(model: nn.Module, dataloader: DeviceDataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model inference on dataloader."""
    ids, preds = [], []
    model = model.to(device)
    model = model.eval()
    
    for x, y in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            p = model(x).clip(0, 1)

        for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p.cpu()):
            ids.append(idx[mask])
            preds.append(pi[mask[:pi.shape[0]]])
    
    ids_tensor = torch.concat(ids)
    preds_tensor = torch.concat(preds)
    
    return ids_tensor, preds_tensor


def main():
    """Main function for RNA structure prediction."""
    # Parse command line arguments
    args = parse_args()
    
    # Set global BPP directory
    global BPP_ROOT_DIR
    BPP_ROOT_DIR = Path(args.bpp_path)
    
    # Setup paths
    model_path = args.model_path
    name = "_".join(model_path.split("/")[-2:]).removesuffix(".pth")
    submit_path = Path(args.out_path) / f'submit_{name}.parquet'
    
    print(f"Model path: {model_path}")
    print(f"Output path: {submit_path}")
    
    # Check if output file already exists
    if submit_path.exists():
        print("Output file already exists")
        raise FileExistsError("Submit path should not lead to existing file.")
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(args.test_path)
    
    # Load bracket structures
    print("Loading bracket structures...")
    bracket_data = load_bracket_structures(args.brackets)
    
    # Initialize model
    print("Initializing model...")
    model = RNAdjNetBrk(
        positional_embedding=args.pos_embedding,
        brk_names=list(bracket_data.keys()),
        depth=args.num_attn_layers, 
        num_convs=args.num_conv_layers,
        adj_ks=args.adj_ks,
        not_slice=args.not_slice,
        use_se=args.use_se
    )
    
    # Load model weights
    print("Loading model weights...")
    model.load_state_dict(torch.load(model_path, map_location="cpu")['model'])
    model = model.eval()
    
    # Setup device
    device = torch.device(f"cuda:{args.device}")
    
    # Create dataset and dataloader
    print("Setting up dataset and dataloader...")
    dataset = RNA_Dataset_Test(test_df, seq_structs=bracket_data)
    dataloader = DeviceDataLoader(
        DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            drop_last=False, 
            num_workers=args.num_workers, 
            persistent_workers=True
        ), 
        device
    )
    
    # Run inference
    print("Running inference...")
    ids, preds = run_inference(model, dataloader, device)
    
    # Create and save predictions dataframe
    print("Creating prediction dataframe...")
    results_df = pd.DataFrame({
        'id': ids.numpy(), 
        'reactivity_DMS_MaP': preds[:, 1].numpy(), 
        'reactivity_2A3_MaP': preds[:, 0].numpy()
    })
    
    print("Sample of predictions:")
    print(results_df.head())
    
    # Save results
    print(f"Saving predictions to {submit_path}...")
    os.makedirs(Path(args.out_path), exist_ok=True)
    results_df.to_parquet(submit_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()