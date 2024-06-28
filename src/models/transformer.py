"""
Inspired from https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn

from .kv_caching import KeysValues, KVCache


@dataclass
class TransformerConfig:

    tokens_per_block: int
    max_blocks: int

    num_layers: int
    num_heads: int
    embed_dim: int

    attention: str

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        self.emb_drop = nn.Dropout(config.embed_pdrop)
        self.ln = nn.LayerNorm(config.embed_dim)

        assert config.attention in ('causal', 'block_causal')
        k, m = config.tokens_per_block, config.max_blocks
        mask_sa = torch.tril(torch.ones(k * m, k * m))
        if config.attention == 'block_causal':
            mask_sa = torch.max(mask_sa, torch.block_diag(*[torch.ones(k, k) for _ in range(m)]))
        mask_sa = mask_sa.bool()

        self.blocks = nn.ModuleList([EncoderLayer(config, mask_sa) for _ in range(config.num_layers)])
        self.keys_values = None

    @property
    def num_blocks_left_in_kv_cache(self) -> float:
        assert self.keys_values is not None
        return (self.config.max_tokens - self.keys_values.size) / self.config.tokens_per_block

    def reset_kv_cache(self, n: int) -> None:
        device = self.ln.weight.device
        self.keys_values = KeysValues(n, self.config.max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, x: torch.FloatTensor, use_kv_cache: bool = False) -> torch.FloatTensor:
        assert x.ndim == 3 and x.size(2) == self.config.embed_dim   # (B, TK, E)

        prev_steps = self.keys_values.size if use_kv_cache else 0
        inputs = x + self.pos_emb(prev_steps + torch.arange(x.size(1), device=x.device))

        y = self.emb_drop(inputs)
        for i, block in enumerate(self.blocks):
            y = block(y, self.keys_values[i] if use_kv_cache else None)
        y = self.ln(y)

        return y


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig, mask_sa: torch.LongTensor) -> None:
        super().__init__()
        self.sa = SelfAttentionLayer(config, mask=mask_sa)
        self.mlp = MLPLayer(config)

    def forward(self, x: torch.FloatTensor, kv_cache: Optional[KVCache] = None) -> torch.FloatTensor:
        return self.mlp(self.sa(x, kv_cache))   


class MLPLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        return inputs + self.mlp(self.ln(inputs)) 


class SelfAttentionLayer(nn.Module):
    def __init__(self, config: TransformerConfig, mask: torch.BoolTensor) -> None:
        super().__init__()
        self.register_buffer('mask', mask)
        self.ln = nn.LayerNorm(config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attention = Attention(config)

    def forward(self, inputs: torch.FloatTensor, kv_cache: Optional[KVCache] = None) -> torch.FloatTensor:
        B, T, C = inputs.size()
        if kv_cache is not None:
            b, L, c = kv_cache.shape
            assert b == B and c == C
        else:
            L = 0

        x = self.ln(inputs)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        y = inputs + self.attention(q, k, v, self.mask[L:L + T, :L + T])

        return y


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.attn_pdrop = config.attn_pdrop
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, q: torch.FloatTensor, k: torch.FloatTensor, v: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        assert mask.size(0) == q.size(1) and mask.size(1) == k.size(1)

        q = rearrange(q, 'b q (h e) -> b h q e', h=self.num_heads)
        k = rearrange(k, 'b k (h e) -> b h k e', h=self.num_heads)
        v = rearrange(v, 'b k (h d) -> b h k d', h=self.num_heads)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_pdrop, is_causal=False) if q.size(2) != 0 else q.new_empty(*q.shape[:-1], v.size(-1))

        y = rearrange(y, 'b h q d -> b q (h d)')
        y = self.resid_drop(self.proj(y))

        return y
