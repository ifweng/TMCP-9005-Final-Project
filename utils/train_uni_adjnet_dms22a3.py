import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, RandomSampler
from x_transformers import TransformerWrapper, Encoder
import numpy as np 
from sklearn.model_selection import KFold
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple, Optional, Union, Any
import os
import gc
from fastai.vision.all import *
import json
import random
from einops import rearrange
from functools import partial
from xpos_relative_position import XPOS


# Utility functions
def exists(val: Any) -> bool:
    """Check if a value exists (is not None)."""
    return val is not None


def pad_at_dim(t: torch.Tensor, pad: Tuple[int, int], dim: int = -1, value: float = 0.) -> torch.Tensor:
    """Pad a tensor at a specific dimension."""
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def seed_everything(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def val_to(x: Any, device: str = "cuda") -> Any:
    """Move a value or list of values to the specified device."""
    if isinstance(x, list):
        return [val_to(z, device) for z in x]
    return x.to(device)


def dict_to(x: Dict[str, Any], device: str = 'cuda') -> Dict[str, Any]:
    """Move all values in a dictionary to the specified device."""
    return {k: val_to(x[k], device) for k in x}


def to_device(x: Tuple[Dict[str, Any], ...], device: str = 'cuda') -> Tuple[Dict[str, Any], ...]:
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
            torch.empty(self.hidden_dim, self.hidden_dim)  # Output projection weights
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
        
        attention = attention + self.gamma * adj
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


class ResConv2dSimple(nn.Module):
    """Residual 2D convolutional layer."""
    def __init__(
        self, 
        in_c: int, 
        out_c: int,
        kernel_size: int = 7
    ):  
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c, 
                kernel_size=kernel_size, 
                padding="same", 
                bias=False
            ),
            nn.BatchNorm2d(out_c),  # maybe batchnorm
            nn.GELU(),
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
            ) for i in range(num_layers)]
        )
        self.conv_layers = nn.ModuleList()
        for i in range(num_adj_convs):
            self.conv_layers.append(
                ResConv2dSimple(
                    in_c=1 if i == 0 else num_heads,
                    out_c=num_heads,
                    kernel_size=ks
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
            ks=adj_ks
        )
        
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )
        
        self.struct_embeds = nn.ModuleDict()
        
        # Commented out in original code
        # if self.brk_names is not None:
        #     for method in self.brk_names:
        #         emb = nn.Embedding(brk_symbols + 3, dim)
        #         self.struct_embeds[method] = emb
        #     self.struct_embeds_weights = torch.nn.Parameter(torch.ones(len(brk_names)))
            
        self.is_good_embed = nn.Embedding(2, dim)
        self.react_emb = nn.Conv1d(1, dim, kernel_size=1)
        self.react_missing_emb = nn.Embedding(2, dim)
            
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
        
        rmask = x0["react"].isnan().int()
        r_missing = self.react_missing_emb(rmask)[:, :Lmax] if self.slice_tokens else self.react_missing_emb(rmask)
        
        react = x0["react"].unsqueeze(1)  # B 1 L
        react = react.clip(0, 1).nan_to_num(-1.0)
        react = self.react_emb(react).permute(0, 2, 1)  # B L E    
        react = react[:, :Lmax] if self.slice_tokens else react
    
        x = e
        is_good = x0['is_good']
        e_is_good = self.is_good_embed(is_good)  # B E
        e_is_good = e_is_good.unsqueeze(1)  # B 1 E
        x = x + e_is_good + react + r_missing
        
        # Commented out in original code
        # if self.brk_names is not None:
        #     for ind, method in enumerate(self.brk_names):
        #         st = x0[method]
        #         if self.slice_tokens:
        #             st = st[:, :Lmax]
        #         st_embed = self.struct_embeds[method](st)
        #         x = x + st_embed * self.struct_embeds_weights[ind]
                
        x = self.transformer(x, adj, mask=mask)
        
        x = self.proj_out(x).squeeze(-1)
   
        return x


class BPPFeatures:
    """Base Pair Probability Features from memory-mapped file."""
    LMAX: ClassVar[int] = 206

    def __init__(self, index_path: str, mempath: str):
        self.index = self.read_index(index_path)
        self.storage = self.read_memmap(mempath, len(self.index))
        
    @classmethod
    def read_index(cls, index_path: str) -> Dict[str, int]:
        with open(index_path) as inp:
            ids = [line.strip() for line in inp]
        index = {seqid: i for i, seqid in enumerate(ids)}
        return index
    
    @classmethod
    def read_memmap(cls, memmap_path: str, index_len: int) -> np.ndarray:
        storage = np.memmap(
            memmap_path, 
            dtype=np.float32,
            mode='r', 
            offset=0,
            shape=(index_len, cls.LMAX, cls.LMAX),
            order='C'
        )
        return storage
    
    def __getitem__(self, seqid: str) -> np.ndarray:
        ind = self.index[seqid]
        return self.storage[ind]


class MISSING:
    """Placeholder for required parameters."""
    pass


def load_bpp(seq_id: str, maxL: int) -> np.ndarray:
    """Load base pair probability matrix for a sequence."""
    path = BPP_ROOT_DIR / f"{seq_id}.npy"
    mat = np.load(path)
    dif = maxL - mat.shape[0]
    res = np.pad(mat, ((0, dif), (0, dif)))
    return res


class RNA_Dataset(Dataset):
    """Dataset for RNA sequences with reactivity data."""
    def __init__(
        self, 
        df: pd.DataFrame,
        seq_structs: Dict[str, Dict[str, str]],
        split_type: str,
        Lmax: int,
        use_shift: bool,
        use_reverse: bool,
        train_threshold: Any = MISSING,
        mode: str = 'train', 
        pred_mode: str = "dms_2a3",
        seed: int = 2023, 
        fold: int = 0, 
        nfolds: int = 4
    ):
        if mode == "train" and train_threshold is MISSING:
            raise Exception("Train threshold should be specified for train mode")
            
        self.seq_map = {
            'A': 0, 'C': 1, 'G': 2, 'U': 3, 
            "START": 4, "END": 5, "EMPTY": 6
        }
        self.brk_map = {
            "(": 0, ")": 1, "[": 2, "]": 3, "{": 4, "}": 5,
            "<": 6, ">": 7, ".": 8, "START": 9, "END": 10, "EMPTY": 11
        }
        
        assert mode in ('train', 'eval')
        self.pred_mode = pred_mode
        self.Lmax = Lmax
        self.seq_structs = seq_structs
        df['L'] = df.sequence.apply(len)

        # Split data according to split type
        if split_type == "kfold":
            df_2A3 = df.loc[df.experiment_type == '2A3_MaP']
            df_DMS = df.loc[df.experiment_type == 'DMS_MaP']
            split = list(KFold(n_splits=nfolds, random_state=seed, 
                    shuffle=True).split(df_2A3))[fold][0 if mode == 'train' else 1]
            df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
            df_DMS = df_DMS.iloc[split].reset_index(drop=True)
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
        
        # Filter data based on signal quality
        if mode == "eval":
            print("Keeping only clean data for validation")
            m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
            df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            df_DMS = df_DMS.loc[m].reset_index(drop=True)
        elif mode == "train":
            if train_threshold is not None:
                print(f"Using threshold {train_threshold}")
                if train_threshold == "clean":
                    m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
                else:
                    try:
                        train_threshold = float(train_threshold)
                    except ValueError:
                        raise Exception("Threshold must be None, float or clean")
                    m = np.logical_and(
                        df_2A3['signal_to_noise'].values >= train_threshold,  
                        df_DMS['signal_to_noise'].values >= train_threshold
                    )
                df_2A3 = df_2A3.loc[m].reset_index(drop=True)
                df_DMS = df_DMS.loc[m].reset_index(drop=True)
            else:
                print(f"Using no threshold")
        else:
            raise Exception(f"Wrong mode: {mode}")
            
        print("2A3 shape after filter", df_2A3.shape, "threshold", train_threshold, "split", split_type, "mode", mode)
        
        # Store data attributes
        self.sid = df_2A3['sequence_id'].values
        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values
        
        # Extract reactivity data
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if 'reactivity_error_0' in c]].values
        
        # Signal quality indicators
        self.is_good = ((df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)) * 1
        self.sn_2A3 = df_2A3['SN_filter'].values 
        self.sn_DMS = df_DMS['SN_filter'].values
        
        # Calculate sample weights based on signal-to-noise ratio
        sn = (df_2A3['signal_to_noise'].values + df_DMS['signal_to_noise'].values) / 2
        sn = torch.from_numpy(sn)
        self.weights = 0.5 * torch.clamp_min(torch.log(sn + 1.01), 0.01)
        
        self.mode = mode
        
        # Data augmentation options
        self.use_shift = use_shift
        if self.use_shift:
            print("Use shifting")
        self.use_reverse = use_reverse
        if self.use_reverse:
            print("Use reverse")
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.seq)  
    
    def _process_seq(self, rawseq: str, shift: int) -> Tuple[torch.Tensor, int, int]:
        """Process a raw RNA sequence into token indices."""
        seq = [self.seq_map['EMPTY'] for i in range(shift)]
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
    
    def _process_brk(self, rawbrk: str, shift: int) -> torch.Tensor:
        """Process a raw bracket structure into token indices."""
        brk = [self.brk_map['EMPTY'] for i in range(shift)]
        brk.append(self.brk_map['START'])
        brk.extend(self.brk_map[b] for b in rawbrk)
        brk.append(self.brk_map['END'])
        
        for i in range(len(brk), self.Lmax + 2):
            brk.append(self.brk_map['EMPTY'])
         
        brk_array = np.array(brk)
        brk_tensor = torch.from_numpy(brk_array)
        return brk_tensor
    
    def get_shift(self, seqL: int) -> int:
        """Get random shift value for data augmentation."""
        if not self.use_shift:
            return 0
        if self.mode == "eval":
            return 0
        
        dif = self.Lmax - seqL 
        shift = torch.randint(low=0, high=dif+1, size=(1,)).item()  # high is not included
        return shift
    
    def get_to_rev(self) -> bool:
        """Determine whether to reverse the sequence for data augmentation."""
        if not self.use_reverse:
            return False
        if self.mode == "eval":
            return False
        
        return torch.rand(1).item() > 0.5
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get a dataset item with optional data augmentation."""
        seq = self.seq[idx]
        real_seq_L = len(seq)
        
        # Apply random shift for data augmentation
        shift = self.get_shift(real_seq_L)
        
        lbord = 1 + shift
        rbord = self.Lmax + 1 - real_seq_L - shift
        
        # Process sequence
        seq_int, start_loc, end_loc = self._process_seq(seq, shift)
        mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)
        mask[start_loc + 1:end_loc] = True  # not including START and END
      
        forward_mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)  # START, seq, END
        forward_mask[start_loc:end_loc + 1] = True  # including START and END
        
        # Process reactivity data
        react = np.stack([
            self.react_2A3[idx][:real_seq_L],
            self.react_DMS[idx][:real_seq_L]
        ], -1)
        react = np.pad(
            react, 
            ((lbord, rbord), (0, 0)), 
            constant_values=np.nan
        )
        
        react = torch.from_numpy(react)
         
        # Select input and output reactivity based on prediction mode
        if self.pred_mode == "dms_2a3":
            x_react = react[:, 1].clip(0, 1)  # DMS
            y_react = react[:, 0]  # 2A3
        elif self.pred_mode == "2a3_dms":
            x_react = react[:, 0].clip(0, 1)  # 2A3
            y_react = react[:, 1]  # DMS
        
        # Build input dictionary
        X = {
            'seq_int': seq_int,
            'mask': mask, 
            "forward_mask": forward_mask,
            'is_good': self.is_good[idx],
            'react': x_react.float()
        }
        
        # Load adjacent matrix for this sequence
        sid = self.sid[idx]
        # Commented out in original code
        # for method, structs in self.seq_structs.items():
        #     brk = structs[sid]
        #     X[method] = self._process_brk(brk, shift)
        
        # Load and process adjacency matrix
        adj = load_bpp(sid, self.Lmax)[:real_seq_L, :real_seq_L]
        adj = np.pad(adj, ((lbord, rbord), (lbord, rbord)), constant_values=0)
        adj = torch.from_numpy(adj).float()
        X['adj'] = adj
        
        # Build output dictionary
        y = {
            'react': y_react.float(), 
            'mask': mask
        }
        
        # Apply sequence reversal augmentation if needed
        to_rev = self.get_to_rev()
        if to_rev:
            X_rev = {}
            for key, value in X.items():
                if key == "is_good":
                    X_rev[key] = value
                elif key == "seq_int":
                    X_rev[key] = value.flip(dims=[0])
                elif key == "mask":
                    X_rev[key] = value.flip(dims=[0])
                elif key == "forward_mask":
                    X_rev[key] = value.flip(dims=[0])
                elif key == "adj":
                    # SxS
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
                elif key in ("react", "react_err"):
                    y_rev[key] = value.flip(dims=[0])
                else:
                    raise Exception(f"No reverse process for key {key}")
            y = y_rev
        
        return X, y


class DeviceDataLoader:
    """DataLoader wrapper that moves batches to a specified device."""
    def __init__(self, dataloader: DataLoader, device: Union[str, torch.device] = 'cuda'):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self) -> int:
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)


def loss(pred: torch.Tensor, target: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Calculate L1 loss for RNA reactivity prediction."""
    p = pred[target['mask'][:, :pred.shape[1]]]
    y = target['react'][target['mask']].clip(0, 1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    
    return loss


class MAE(Metric):
    """Mean Absolute Error metric for RNA reactivity prediction."""
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.x, self.y = [], []
        
    def accumulate(self, learn):
        x = learn.pred[learn.y['mask'][:, :learn.pred.shape[1]]]
        y = learn.y['react'][learn.y['mask']].clip(0, 1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x, y = torch.cat(self.x, 0), torch.cat(self.y, 0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss


def parse_args():
    """Parse command line arguments for RNA model training."""
    parser = argparse.ArgumentParser(description="RNA Structure Prediction Model Training")
    parser.add_argument("--bpp_path",
                        required=True,
                        type=str,
                        help="Path to base pair probability files")
    parser.add_argument("--train_path", 
                        required=True,
                        type=str,
                        help="Path to training data parquet file")
    parser.add_argument("--brackets", 
                        required=False,
                        default=[],
                        type=str, 
                        nargs='+',
                        help="Path to bracket structure JSON files")
    parser.add_argument("--out_path", 
                        required=True,
                        type=str,
                        help="Output directory for model weights")
    parser.add_argument("--num_workers",
                        default=32,
                        type=int,
                        help="Number of workers for data loading")
    parser.add_argument("--nfolds",
                        default=4,
                        type=int,
                        help="Number of folds for cross-validation")
    parser.add_argument("--fold",
                        default=0,
                        type=int,
                        help="Which fold to train on")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="Batch size for training")
    parser.add_argument("--device",
                        default=0,
                        type=int,
                        help="CUDA device index")
    parser.add_argument("--seed",
                        default=2023,
                        type=int,
                        help="Random seed for reproducibility")
    parser.add_argument("--epoch",
                        default=200,
                        type=int,
                        help="Number of epochs for main training")
    parser.add_argument("--lr_max",
                        default=2.5e-3,
                        type=float,
                        help="Maximum learning rate for one-cycle policy")
    parser.add_argument("--wd",
                        default=0.05,
                        type=float,
                        help="Weight decay for main training")
    parser.add_argument("--pct_start",
                        default=0.05,
                        type=float,
                        help="Percentage of cycle spent increasing learning rate")
    parser.add_argument("--gradclip",
                        default=1.0,
                        type=float,
                        help="Gradient clipping value")
    parser.add_argument("--num_attn_layers",
                        default=12,
                        type=int,
                        help="Number of attention layers in model")
    parser.add_argument("--num_conv_layers",
                        default=12,
                        type=int,
                        help="Number of convolutional layers in model")
    parser.add_argument("--adj_ks",
                        required=True, 
                        type=int,
                        help="Kernel size for adjacency convolutions")
    parser.add_argument("--split",
                        choices=["kfold", "length", "sim", "order", "table"],
                        default="kfold",
                        type=str,
                        help="Data split method")
    parser.add_argument("--batch_cnt",
                        default=1791,
                        type=int,
                        help="Number of batches per epoch")
    parser.add_argument("--sgd_lr",
                        default=5e-5,
                        type=float,
                        help="Learning rate for SGD fine-tuning")
    parser.add_argument("--sgd_epochs",
                        default=25,
                        type=int,
                        help="Number of epochs for SGD fine-tuning")
    parser.add_argument("--sgd_batch_cnt",
                        default=500,
                        type=int,
                        help="Number of batches per epoch for SGD fine-tuning")
    parser.add_argument("--sgd_wd",
                        default=0.05,
                        type=float,
                        help="Weight decay for SGD fine-tuning")
    parser.add_argument("--Lmax",
                        default=206,
                        type=int,
                        help="Maximum sequence length")
    parser.add_argument("--use_shift",
                        action="store_true",
                        help="Use sequence shifting augmentation")
    parser.add_argument("--pos_embedding",
                        choices=['xpos', 'dyn', 'alibi'],
                        required=True,
                        help="Type of positional embedding")
    parser.add_argument("--train_threshold",
                        default=None,
                        help="Signal-to-noise threshold for training data")
    parser.add_argument("--pretrained_model",
                        default=None,
                        help="Path to pretrained model weights")
    parser.add_argument("--not_slice",
                        action="store_true",
                        help="Don't slice padding tokens")
    parser.add_argument("--no_weights",
                        action="store_true",
                        help="Don't use weighted sampling")
    parser.add_argument("--pred_mode",
                        choices=["dms_2a3", "2a3_dms"],
                        required=True,
                        help="Prediction mode (which reactivity to predict)")
    
    return parser.parse_args()


def load_bracket_structures(bracket_paths: List[str]) -> Dict[str, Dict[str, str]]:
    """Load bracket structure files from paths."""
    bracket_data = {}
    for br_path in bracket_paths:
        name = Path(br_path).name.replace(".json", "")
        with open(br_path) as inp:
            data = json.load(inp)
            bracket_data[name] = data
    
    return bracket_data


def setup_training_directories(output_path: str) -> Tuple[Path, Path, Path]:
    """Setup directories for training output."""
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    first_model_dir = output_dir / "fst_model" / "pos_aug"
    first_model_dir.mkdir(parents=True, exist_ok=True)
    
    sgd_model_dir = output_dir / "fst_model" / "pos_aug_sgd"
    sgd_model_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, first_model_dir, sgd_model_dir


def create_dataloaders(
    df: pd.DataFrame,
    bracket_data: Dict[str, Dict[str, str]],
    args: argparse.Namespace,
    device: torch.device
) -> Tuple[DeviceDataLoader, DeviceDataLoader, RNA_Dataset, RNA_Dataset]:
    """Create training and validation dataloaders."""
    # Create training dataset
    print("Loading training dataset...")
    train_dataset = RNA_Dataset(
        df, 
        use_shift=args.use_shift,
        Lmax=args.Lmax, 
        train_threshold=args.train_threshold,
        seq_structs=bracket_data, 
        split_type=args.split,
        use_reverse=False,
        mode='train', 
        fold=args.fold, 
        nfolds=args.nfolds,
        pred_mode=args.pred_mode,
        seed=args.seed
    )
    
    # Create validation dataset
    print("Loading validation dataset...")
    val_dataset = RNA_Dataset(
        df, 
        use_shift=args.use_shift,
        use_reverse=False,
        Lmax=args.Lmax, 
        seq_structs=bracket_data, 
        split_type=args.split,
        mode='eval',
        fold=args.fold,
        nfolds=args.nfolds,
        pred_mode=args.pred_mode,
        seed=args.seed
    )
    
    # Setup samplers for balanced training
    if args.no_weights:
        train_sampler = RandomSampler(
            train_dataset, 
            replacement=True, 
            num_samples=args.batch_size * args.batch_cnt
        )
    else:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.weights, 
            num_samples=args.batch_size * args.batch_cnt
        )
    
    # Create dataloaders
    train_dataloader = DeviceDataLoader(
        DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            sampler=train_sampler, 
            num_workers=args.num_workers,
            persistent_workers=True
        ), 
        device
    )
    
    val_dataloader = DeviceDataLoader(
        DataLoader(
            val_dataset, 
            batch_size=args.batch_size * 4,
            num_workers=args.num_workers,  
            persistent_workers=True
        ), 
        device
    )
    
    return train_dataloader, val_dataloader, train_dataset, val_dataset


def create_model(args: argparse.Namespace, bracket_data: Dict[str, Dict[str, str]], device: torch.device) -> nn.Module:
    """Create and initialize the RNA model."""
    model = RNAdjNetBrk(
        positional_embedding=args.pos_embedding,
        brk_names=list(bracket_data.keys()),
        depth=args.num_attn_layers, 
        num_convs=args.num_conv_layers,
        adj_ks=args.adj_ks,
        not_slice=args.not_slice
    )
    
    # Load pretrained weights if specified
    if args.pretrained_model is not None:
        print(f"Loading pretrained model from {args.pretrained_model}")
        model.load_state_dict(
            torch.load(args.pretrained_model, map_location="cpu")['model']
        )
    
    return model.to(device)


def main():
    """Main function for RNA model training."""
    # Parse command line arguments
    args = parse_args()
    print(args)
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Set global BPP directory
    global BPP_ROOT_DIR
    BPP_ROOT_DIR = Path(args.bpp_path)
    
    # Load training data
    print("Loading training data...")
    df = pd.read_parquet(args.train_path)
    
    # Load bracket structure data
    print("Loading bracket structures...")
    bracket_data = load_bracket_structures(args.brackets)
    
    # Setup output directories
    output_dir, first_model_dir, sgd_model_dir = setup_training_directories(args.out_path)
    
    # Setup device
    device = torch.device(f"cuda:{args.device}")
    
    # Process only the specified fold
    fold = args.fold
    print(f"Processing fold {fold}/{args.nfolds}")
    
    # Create dataloaders
    train_dataloader, val_dataloader, train_dataset, val_dataset = create_dataloaders(
        df, bracket_data, args, device
    )
    
    # Create fastai DataLoaders
    data = DataLoaders(train_dataloader, val_dataloader)
    
    # Clean up memory
    gc.collect()
    
    # Create model
    model = create_model(args, bracket_data, device)
    print(model)
    
    # Setup callbacks for training
    callbacks = [
        GradientClip(args.gradclip),
        CSVLogger(fname=str(first_model_dir / "loss.csv")),
        SaveModelCallback(
            monitor='valid_loss',
            fname='model', 
            with_opt=True
        )
    ]
    
    # Create fastai Learner for main training
    learner = Learner(
        data, 
        model, 
        loss_func=loss,
        model_dir=first_model_dir,
        cbs=callbacks,
        metrics=[MAE()]
    ).to_fp16()  # fp16 doesn't help at P100 but gives x1.6-1.8 speedup at modern hardware
    
    # Run main training with one-cycle policy
    print("Starting main training phase...")
    learner.fit_one_cycle(
        args.epoch,
        lr_max=args.lr_max,
        wd=args.wd,
        pct_start=args.pct_start
    )
    
    # Save model weights
    torch.save(
        learner.model.state_dict(),
        output_dir / f'model_{fold}.pth'
    )
    
    # Clean up memory
    gc.collect()
    
    # Create SGD fine-tuning dataloader
    print("Setting up SGD fine-tuning phase...")
    if args.no_weights:
        sgd_train_sampler = RandomSampler(
            train_dataset, 
            replacement=True, 
            num_samples=args.batch_size * args.sgd_batch_cnt
        )
    else:
        sgd_train_sampler = WeightedRandomSampler(
            weights=train_dataset.weights, 
            num_samples=args.batch_size * args.sgd_batch_cnt
        )

    sgd_train_dataloader = DeviceDataLoader(
        DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            sampler=sgd_train_sampler, 
            num_workers=args.num_workers,
            persistent_workers=True
        ), 
        device
    )
        
    sgd_val_dataloader = DeviceDataLoader(
        DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,  
            persistent_workers=True
        ), 
        device
    )
    
    sgd_data = DataLoaders(sgd_train_dataloader, sgd_val_dataloader)
    
    # Setup callbacks for SGD fine-tuning
    sgd_callbacks = [
        GradientClip(args.gradclip),
        CSVLogger(fname=str(sgd_model_dir / "loss.csv")),
        SaveModelCallback(
            monitor='valid_loss',
            fname='model', 
            with_opt=True
        )
    ]
    
    # Create fastai Learner for SGD fine-tuning
    sgd_learner = Learner(
        sgd_data,
        model,
        model_dir=sgd_model_dir,
        lr=args.sgd_lr,
        opt_func=partial(OptimWrapper, opt=torch.optim.SGD),
        loss_func=loss,
        cbs=sgd_callbacks,
        metrics=[MAE()]
    ).to_fp16()
    
    # Run SGD fine-tuning
    print("Starting SGD fine-tuning phase...")
    sgd_learner.fit(
        args.sgd_epochs, 
        wd=args.sgd_wd
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main()