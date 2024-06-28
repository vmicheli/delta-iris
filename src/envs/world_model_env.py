from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision

from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import compute_softmax_over_buckets, symexp


@dataclass
class WorldModelEnvOutput:
    frames: torch.FloatTensor
    wm_output_sequence: torch.FloatTensor


class WorldModelEnv:
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:
        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()
        self.env = env

        self.obs = None
        self.x = None
        self.last_latent_token_emb = None

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = rearrange(torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device), 'c h w -> 1 1 c h w')
        act = torch.empty(obs.size(0), 0, dtype=torch.long, device=self.device)
        return self.reset_from_past(obs, act)

    @torch.no_grad()
    def reset_from_past(self, obs: torch.FloatTensor, act: torch.LongTensor) -> Tuple[WorldModelEnvOutput, WorldModelEnvOutput]:
        self.obs = obs[:, -1:]
        self.x = None
        self.last_latent_token_emb = None
        self.world_model.transformer.reset_kv_cache(n=obs.size(0))

        latent_tokens = self.tokenizer.burn_in(obs, act)
        wm_output_sequence = self.world_model.burn_in(obs, act, latent_tokens, use_kv_cache=True)

        obs_burn_in_policy = WorldModelEnvOutput(obs[:, :-1], wm_output_sequence[:, :-1]) 
        first_obs = WorldModelEnvOutput(obs[:, -1:], wm_output_sequence[:, -1:])

        return obs_burn_in_policy, first_obs  

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor]) -> Tuple[Optional[WorldModelEnvOutput], float, float, None]:
        if self.world_model.transformer.num_blocks_left_in_kv_cache <= 1:
            self.world_model.transformer.reset_kv_cache(n=self.obs.size(0))
            self.last_latent_token_emb = None

        wm_output_sequence = []

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long).reshape(-1, 1).to(self.device)
        a = self.world_model.act_emb(action)

        if self.last_latent_token_emb is None:
            if self.x is None:
                outputs_wm = self.world_model(a, use_kv_cache=True)
            else:
                outputs_wm = self.world_model(torch.cat((self.x, a), dim=1), use_kv_cache=True)
        else:
            outputs_wm = self.world_model(torch.cat((self.last_latent_token_emb, self.x, a), dim=1), use_kv_cache=True)

        wm_output_sequence.append(outputs_wm.output_sequence)

        if self.world_model.config.two_hot_rews:
            reward = symexp(compute_softmax_over_buckets(outputs_wm.logits_rewards))
        else:
            reward = Categorical(logits=outputs_wm.logits_rewards).sample().float() - 1
        reward = reward.flatten().cpu().numpy()
        done = Categorical(logits=outputs_wm.logits_ends).sample().bool().flatten().cpu().numpy()

        latent_tokens = []

        latent_token = Categorical(logits=outputs_wm.logits_latents).sample()
        latent_tokens.append(latent_token)

        for _ in range(self.tokenizer.config.num_tokens - 1):
            latent_token_emb = self.world_model.latents_emb(latent_token)
            outputs_wm = self.world_model(latent_token_emb, use_kv_cache=True)
            wm_output_sequence.append(outputs_wm.output_sequence)

            latent_token = Categorical(logits=outputs_wm.logits_latents).sample()
            latent_tokens.append(latent_token)

        self.last_latent_token_emb = self.world_model.latents_emb(latent_token)

        q = self.tokenizer.quantizer.embed_tokens(torch.stack(latent_tokens, dim=-1))
        self.obs = self.tokenizer.decode(
            self.obs,
            action,
            rearrange(q, 'b t (h w) (k l e) -> b t e (h k) (w l)', h=self.tokenizer.tokens_grid_res, k=self.tokenizer.token_res, l=self.tokenizer.token_res),
            should_clamp=True
        )

        self.x = rearrange(self.world_model.frame_cnn(self.obs), 'b 1 k e -> b k e')

        obs = WorldModelEnvOutput(frames=self.obs, wm_output_sequence=torch.cat(wm_output_sequence, dim=1))

        return obs, reward, done, None

    @torch.no_grad()
    def render(self):
        return Image.fromarray(rearrange(self.obs, '1 1 c h w -> h w c').mul(255).cpu().numpy().astype(np.uint8))

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        raise NotImplementedError
