from dataclasses import dataclass
import math
from typing import Dict, Tuple

from einops import rearrange
import torch
import torch.nn as nn

from ..convnet import FrameCnnConfig, FrameEncoder, FrameDecoder
from data import Batch
from .quantizer import Quantizer, QuantizerOutput
from utils import init_weights, LossWithIntermediateLosses


@dataclass
class TokenizerConfig:
    image_channels: int
    image_size: int
    num_actions: int
    num_tokens: int
    decoder_act_channels: int
    codebook_size: int
    codebook_dim: int
    max_codebook_updates_with_revival: int
    encoder_config: FrameCnnConfig
    decoder_config: FrameCnnConfig
    frame_cnn_config: FrameCnnConfig


class Tokenizer(nn.Module):
    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        self.config = config

        self.latent_res = config.image_size // 2 ** sum(config.encoder_config.down)
        self.tokens_grid_res = int(math.sqrt(config.num_tokens))
        self.token_res = self.latent_res // self.tokens_grid_res

        self.encoder_act_emb = nn.Embedding(config.num_actions, config.image_size ** 2)
        self.decoder_act_emb = nn.Embedding(config.num_actions, config.decoder_act_channels * self.latent_res ** 2)

        self.quantizer = Quantizer(
            config.codebook_size, config.codebook_dim,
            input_dim=config.encoder_config.latent_dim * self.token_res ** 2,
            max_codebook_updates_with_revival=config.max_codebook_updates_with_revival
        )

        self.encoder = FrameEncoder(config.encoder_config)
        self.decoder = FrameDecoder(config.decoder_config)
        self.frame_cnn = FrameEncoder(config.frame_cnn_config)

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x1: torch.FloatTensor, a: torch.LongTensor, x2: torch.FloatTensor) -> QuantizerOutput:
        z = self.encode(x1, a, x2)
        z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)', h=self.tokens_grid_res, w=self.tokens_grid_res)

        return self.quantizer(z)

    def compute_loss(self, batch: Batch, **kwargs) -> Tuple[LossWithIntermediateLosses, Dict]:
        x1 = batch.observations[:, :-1]
        a = batch.actions[:, :-1]
        x2 = batch.observations[:, 1:]

        quantizer_outputs = self(x1, a, x2)

        r = self.decode(x1, a, rearrange(quantizer_outputs.q, 'b t (h w) (k l e) -> b t e (h k) (w l)', h=self.tokens_grid_res, k=self.token_res, l=self.token_res))
        delta = (x2 - r)
        delta = delta[torch.logical_and(batch.mask_padding[:, 1:], batch.mask_padding[:, :-1])]

        losses = {
            **quantizer_outputs.loss,
            'reconstruction_loss_l1': 0.1 * torch.abs(delta).mean(),
            'reconstruction_loss_l2': delta.pow(2).mean(),
            'reconstruction_loss_l2_worst_pixel': 0.01 * rearrange(delta, 'b c h w -> b (c h w)').pow(2).max(dim=-1)[0].mean(),
        }

        return LossWithIntermediateLosses(**losses), quantizer_outputs.metrics

    def encode(self, x1: torch.FloatTensor, a: torch.LongTensor, x2: torch.FloatTensor) -> torch.FloatTensor:
        a_emb = rearrange(self.encoder_act_emb(a), 'b t (h w) -> b t 1 h w', h=x1.size(3))
        encoder_input = torch.cat((x1, a_emb, x2), dim=2)
        z = self.encoder(encoder_input)

        return z

    def decode(self, x1: torch.FloatTensor, a: torch.LongTensor, q2: torch.FloatTensor, should_clamp: bool = False) -> torch.FloatTensor:
        x1_emb = self.frame_cnn(x1)
        a_emb = rearrange(self.decoder_act_emb(a), 'b t (c h w) -> b t c h w', c=self.config.decoder_act_channels, h=x1_emb.size(3))

        decoder_input = torch.cat((x1_emb, a_emb, q2), dim=2)

        r = self.decoder(decoder_input)
        r = torch.clamp(r, 0, 1).mul(255).round().div(255) if should_clamp else r

        return r

    @torch.no_grad()
    def encode_decode(self, x1: torch.FloatTensor, a: torch.LongTensor, x2: torch.FloatTensor) -> torch.Tensor:
        z = self.encode(x1, a, x2)
        z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)', k=self.token_res, l=self.token_res)
        q = rearrange(self.quantizer(z).q, 'b t (h w) (k l e) -> b t e (h k) (w l)', h=self.tokens_grid_res, k=self.token_res, l=self.token_res)
        r = self.decode(x1, a, q, should_clamp=True)

        return r

    @torch.no_grad()
    def burn_in(self, obs: torch.FloatTensor, act: torch.LongTensor) -> torch.LongTensor: 
        assert obs.size(1) == act.size(1) + 1
        quantizer_output = self(obs[:, :-1], act, obs[:, 1:])

        return quantizer_output.tokens
