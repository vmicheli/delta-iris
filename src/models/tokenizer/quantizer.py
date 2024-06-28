from dataclasses import dataclass
import math
from typing import Dict, Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    q: torch.FloatTensor
    tokens: torch.LongTensor
    loss: Dict[str, torch.FloatTensor]
    metrics: Dict[str, float]


class Quantizer(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int, input_dim: int, max_codebook_updates_with_revival: Optional[int] = None) -> None:
        super().__init__()
        assert math.log2(codebook_size).is_integer()
        self.revival_entropy_threshold = int(math.log2(codebook_size)) - 2
        self.max_codebook_updates_with_revival = max_codebook_updates_with_revival
        self.pre_quant_proj = nn.Linear(input_dim, codebook_dim)
        self.post_quant_proj = nn.Linear(codebook_dim, input_dim)
        codebook = torch.empty(codebook_size, codebook_dim, requires_grad=False).uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.register_buffer('num_codebook_updates', torch.tensor(0))
        self.register_buffer('codebook', codebook)
        self.register_buffer('codewords_freqs', torch.ones(codebook_size).div(codebook_size))

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        z = self.pre_quant_proj(z)
        z = F.normalize(z, dim=-1)
        b, k = z.size(0), z.size(2)
        z = rearrange(z, 'b t k e -> (b t k) e')

        cosine_similarity = torch.einsum('n e, c e -> n c', z, self.codebook)
        tokens = cosine_similarity.argmax(dim=-1)
        q = self.codebook[tokens]

        losses = {'commitment_loss': 0.02 * (z - q.detach()).pow(2).mean()}

        if self.training:
            metrics = {**self.update_codebook(z, tokens), 'codebook_entropy': self.compute_codebook_entropy()}
        else:
            metrics = {}

        q = z + (q - z).detach()
        q = self.post_quant_proj(q)

        q = rearrange(q, '(b t k) e -> b t k e', b=b, k=k)
        tokens = rearrange(tokens, '(b t k) -> b t k', b=b, k=k)

        return QuantizerOutput(q, tokens, losses, metrics)

    @torch.no_grad()
    def update_codebook(self, z: torch.Tensor, tokens: torch.LongTensor) -> None:
        tokens_one_hot = F.one_hot(tokens, self.codebook.size(0)).float()  # (N, C)

        # Update codebook
        counts = tokens_one_hot.sum(dim=0) 
        codebook_update = torch.einsum('n e, n c -> c e', z, tokens_one_hot) / torch.clamp(counts.unsqueeze(-1), min=1)
        codebook_update = F.normalize(codebook_update, dim=-1)
        self.codebook.lerp_(codebook_update, 1 - 0.99)

        # Update counts and revive dead codewords
        freqs = counts / tokens_one_hot.size(0)
        self.codewords_freqs.lerp_(freqs, 1 - 0.98)

        can_revive = (self.compute_codebook_entropy() < 1) or (self.max_codebook_updates_with_revival is None) or (self.num_codebook_updates.item() < self.max_codebook_updates_with_revival) 
        if can_revive and (self.compute_codebook_entropy() < self.revival_entropy_threshold):
            expired = torch.where(self.codewords_freqs < 1 / (10 * self.codewords_freqs.size(0)))[0]
            num_expired = expired.size(0)
            expired = expired[torch.randperm(num_expired)[:z.size(0)]]
            idx_revived = torch.randperm(z.size(0), device=z.device)[:expired.size(0)]
            self.codebook[expired] = z[idx_revived]
            self.codewords_freqs[expired] = 1 / self.codewords_freqs.size(0)
        else:
            num_expired = 0

        self.codebook = F.normalize(self.codebook, dim=-1)

        self.num_codebook_updates += 1
        metrics = {'codewords_revived': num_expired}

        return metrics

    def compute_codebook_entropy(self) -> float:
        probs = self.codewords_freqs[self.codewords_freqs != 0]
        return -(torch.log2(probs) * probs).sum().item()

    @torch.no_grad()
    def embed_tokens(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        return self.post_quant_proj(self.codebook[tokens])
