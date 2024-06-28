from dataclasses import dataclass

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnet import FrameCnnConfig, FrameEncoder
from data import Batch
from .slicer import  Head
from .tokenizer import Tokenizer
from .transformer import TransformerEncoder, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses, symlog, two_hot


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_latents: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


@dataclass
class WorldModelConfig:
    latent_vocab_size: int
    num_actions: int
    image_channels: int
    image_size: int
    latents_weight: float
    rewards_weight: float
    ends_weight: float
    two_hot_rews: bool
    transformer_config: TransformerConfig
    frame_cnn_config: FrameCnnConfig


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = TransformerEncoder(config.transformer_config)

        assert ((config.image_size // 2 ** sum(config.frame_cnn_config.down)) ** 2) * config.frame_cnn_config.latent_dim == config.transformer_config.embed_dim
        self.frame_cnn = nn.Sequential(FrameEncoder(config.frame_cnn_config), Rearrange('b t c h w -> b t 1 (h w c)'), nn.LayerNorm(config.transformer_config.embed_dim))

        self.act_emb = nn.Embedding(config.num_actions, config.transformer_config.embed_dim)
        self.latents_emb = nn.Embedding(config.latent_vocab_size, config.transformer_config.embed_dim)

        act_pattern = torch.zeros(config.transformer_config.tokens_per_block)
        act_pattern[1] = 1
        act_and_latents_but_last_pattern = torch.zeros(config.transformer_config.tokens_per_block) 
        act_and_latents_but_last_pattern[1:-1] = 1

        self.head_latents = Head(
            max_blocks=config.transformer_config.max_blocks,
            block_mask=act_and_latents_but_last_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.transformer_config.embed_dim, config.transformer_config.embed_dim), nn.ReLU(),
                nn.Linear(config.transformer_config.embed_dim, config.latent_vocab_size)
            )
        )

        self.head_rewards = Head(
            max_blocks=config.transformer_config.max_blocks,
            block_mask=act_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.transformer_config.embed_dim, config.transformer_config.embed_dim), nn.ReLU(),
                nn.Linear(config.transformer_config.embed_dim, 255 if config.two_hot_rews else 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config.transformer_config.max_blocks,
            block_mask=act_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.transformer_config.embed_dim, config.transformer_config.embed_dim), nn.ReLU(),
                nn.Linear(config.transformer_config.embed_dim, 2)
            )
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    def forward(self, sequence: torch.FloatTensor, use_kv_cache: bool = False) -> WorldModelOutput:      
        prev_steps = self.transformer.keys_values.size if use_kv_cache else 0
        num_steps = sequence.size(1)

        outputs = self.transformer(sequence, use_kv_cache=use_kv_cache)

        logits_latents = self.head_latents(outputs, num_steps, prev_steps)
        logits_rewards = self.head_rewards(outputs, num_steps, prev_steps)
        logits_ends = self.head_ends(outputs, num_steps, prev_steps)

        return WorldModelOutput(outputs, logits_latents, logits_rewards, logits_ends)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs) -> LossWithIntermediateLosses:
        assert torch.all(batch.ends.sum(dim=1) <= 1)

        with torch.no_grad():
            latent_tokens = tokenizer(batch.observations[:, :-1], batch.actions[:, :-1], batch.observations[:, 1:]).tokens

        b, _, k = latent_tokens.size()

        frames_emb = self.frame_cnn(batch.observations)
        act_tokens_emb = self.act_emb(rearrange(batch.actions, 'b t -> b t 1'))
        latent_tokens_emb = self.latents_emb(torch.cat((latent_tokens, latent_tokens.new_zeros(b, 1, k)), dim=1))
        sequence = rearrange(torch.cat((frames_emb, act_tokens_emb, latent_tokens_emb), dim=2), 'b t p1k e -> b (t p1k) e')
  
        outputs = self(sequence)

        mask = batch.mask_padding

        labels_latents = latent_tokens[mask[:, :-1]].flatten()
        logits_latents = outputs.logits_latents[:, :-k][repeat(mask[:, :-1], 'b t -> b (t k)', k=k)]
        latent_acc = (logits_latents.max(dim=-1)[1] == labels_latents).float().mean()
        labels_rewards = two_hot(symlog(batch.rewards)) if self.config.two_hot_rews else (batch.rewards.sign() + 1).long()

        loss_latents = F.cross_entropy(logits_latents, target=labels_latents) * self.config.latents_weight
        loss_rewards = F.cross_entropy(outputs.logits_rewards[mask], target=labels_rewards[mask]) * self.config.rewards_weight
        loss_ends = F.cross_entropy(outputs.logits_ends[mask], target=batch.ends[mask]) * self.config.ends_weight

        return LossWithIntermediateLosses(loss_latents=loss_latents, loss_rewards=loss_rewards, loss_ends=loss_ends), {'latent_accuracy': latent_acc}

    @torch.no_grad()
    def burn_in(self, obs: torch.FloatTensor, act: torch.LongTensor, latent_tokens: torch.LongTensor, use_kv_cache: bool = False) -> torch.FloatTensor: 
        assert obs.size(1) == act.size(1) + 1 == latent_tokens.size(1) + 1

        x_emb = self.frame_cnn(obs)
        act_emb = rearrange(self.act_emb(act), 'b t e -> b t 1 e')
        q_emb = self.latents_emb(latent_tokens)
        x_a_q = rearrange(torch.cat((x_emb[:, :-1], act_emb, q_emb), dim=2), 'b t k2 e -> b (t k2) e')
        wm_input_sequence = torch.cat((x_a_q, x_emb[:, -1]), dim=1)
        wm_output_sequence = self(wm_input_sequence, use_kv_cache=use_kv_cache).output_sequence

        return wm_output_sequence
