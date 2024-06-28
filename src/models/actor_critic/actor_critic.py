from __future__ import annotations
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F

from .cnn_lstm_actor_critic import CnnLstmActorCritic
from data import Batch
from .utils import compute_lambda_returns, ImagineOutput
from envs.world_model_env import WorldModelEnv
from ..tokenizer import Tokenizer
from ..world_model import WorldModel
from utils import compute_mask_after_first_done, compute_softmax_over_buckets, LossWithIntermediateLosses, symexp, symlog, two_hot


@dataclass
class ActorCriticConfig:
    burn_in_length: int
    imagination_horizon: int
    gamma: float
    lambda_: float
    entropy_weight: float
    two_hot_rets: bool
    model: CnnLstmActorCritic


class ActorCritic(nn.Module):
    def __init__(self, config: ActorCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.model = config.model
        self.target_model = deepcopy(config.model)
        self.target_model.requires_grad_(False)

        self._past_obs = None

    def __repr__(self) -> str:
        return "actor_critic"

    def forward(self):
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def past_obs(self) -> torch.FloatTensor:
        return torch.stack(list(self._past_obs), dim=1)

    def reset(self, n: int) -> None:
        self.model.reset(n)
        self.target_model.reset(n)
        maxlen = self.config.burn_in_length + self.config.imagination_horizon
        self._past_obs = deque(maxlen=maxlen)

    def clear(self) -> None:
        self.model.clear()
        self.target_model.clear()
        self._past_obs = None

    def update_target(self) -> None:
        source_state_dict = self.model.state_dict()
        target_state_dict = self.target_model.state_dict()
        TAU = 0.995
        for key in source_state_dict:
            target_state_dict[key] = source_state_dict[key] * (1 - TAU) + target_state_dict[key] * TAU
        self.target_model.load_state_dict(target_state_dict)

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        self._past_obs.append(obs)

        inputs = self.model.build_present_input_from_past(self.past_obs)        
        outputs = self.model(inputs)

        logits_actions = outputs.logits_actions[:, -1]
        act_token = Categorical(logits=logits_actions / temperature).sample() if should_sample else logits_actions.argmax(dim=-1)
        value = symexp(compute_softmax_over_buckets(outputs.logits_values[:, -1])) if self.config.two_hot_rets else outputs.logits_values[:, -1, 0]  

        return act_token, value

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, **kwargs) -> LossWithIntermediateLosses:
        outputs = self.imagine(batch, tokenizer, world_model)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.target_values,
                ends=outputs.ends,
                value_bootstrap=outputs.value_bootstrap,
                gamma=self.config.gamma,
                lambda_=self.config.lambda_
            )
 
        d = Categorical(logits=outputs.logits_actions)
        log_probs = d.log_prob(outputs.actions)

        mask = compute_mask_after_first_done(outputs.ends)

        loss_actions = torch.mean(-log_probs[mask] * (lambda_returns[mask] - outputs.target_values[mask]))
        if self.config.two_hot_rets:
            loss_values = F.cross_entropy(outputs.logits_values[mask], target=two_hot(symlog(lambda_returns[mask])))
        else:
            loss_values = F.mse_loss(outputs.logits_values[mask], target=lambda_returns[mask])
        entropy = torch.mean(d.entropy()[mask])
        loss_entropy = -self.config.entropy_weight * entropy

        self.update_target()

        metrics = {'policy_entropy': entropy.item()}

        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_entropy=loss_entropy, loss_values=loss_values), metrics

    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel) -> ImagineOutput:
        mask_padding = batch.mask_padding
        assert batch.observations.ndim == 5
        assert mask_padding[:, -1].all()
        assert batch.observations.size(1) == self.config.burn_in_length + 1
        assert world_model.config.transformer_config.max_blocks >= self.config.burn_in_length + self.config.imagination_horizon + 1

        wm_env = WorldModelEnv(tokenizer, world_model, self.model.device)

        all_actions = []
        all_logits_actions = []
        all_logits_values = []
        all_target_logits_values = []
        all_rewards = []
        all_ends = []

        self.reset(n=batch.observations.size(0))

        obs_burn_in_policy, obs_wm_env = wm_env.reset_from_past(batch.observations, batch.actions[:, :-1])
        self.model.burn_in(obs_burn_in_policy)
        self.target_model.burn_in(obs_burn_in_policy)

        all_observations = [obs_burn_in_policy, obs_wm_env]

        for _ in range(self.config.imagination_horizon):
            outputs = self.model(obs_wm_env, use_kv_cache=True)
            action_token = Categorical(logits=outputs.logits_actions).sample()

            with torch.no_grad():
                target_logits_values = self.target_model(obs_wm_env, use_kv_cache=True).logits_values

            obs_wm_env, reward, done, _ = wm_env.step(action_token)

            all_observations.append(obs_wm_env)
            all_actions.append(action_token)
            all_logits_actions.append(outputs.logits_actions)
            all_logits_values.append(outputs.logits_values if self.config.two_hot_rets else outputs.logits_values[:, :, 0])
            all_target_logits_values.append(target_logits_values if self.config.two_hot_rets else target_logits_values[:, :, 0])
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        with torch.no_grad():
            logits_values = self.target_model(obs_wm_env).logits_values
            value_bootstrap = symexp(compute_softmax_over_buckets(logits_values[:, -1])) if self.config.two_hot_rets else logits_values[:, -1, 0]

        self.clear()

        with torch.no_grad():
            target_values = torch.cat(all_target_logits_values, dim=1)
            if self.config.two_hot_rets:
                target_values = symexp(compute_softmax_over_buckets(target_values))

        return ImagineOutput(
            actions=torch.cat(all_actions, dim=1),                     # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),       # (B, T, A)
            logits_values=torch.cat(all_logits_values, dim=1),         # (B, T, 255)
            rewards=torch.cat(all_rewards, dim=1).to(self.device),     # (B, T)
            ends=torch.cat(all_ends, dim=1).long().to(self.device),    # (B, T)
            target_values=target_values,                               # (B, T, 255)
            value_bootstrap=value_bootstrap,                           # (B,)
        )
