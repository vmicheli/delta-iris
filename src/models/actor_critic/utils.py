from dataclasses import dataclass

import torch


@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    logits_values: torch.FloatTensor


@dataclass
class ImagineOutput:
    actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    logits_values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor
    target_values: torch.FloatTensor
    value_bootstrap: torch.FloatTensor


@torch.no_grad()
def compute_lambda_returns(
        rewards: torch.FloatTensor,
        values: torch.FloatTensor,
        ends: torch.LongTensor,
        value_bootstrap: torch.FloatTensor,
        gamma: float,
        lambda_: float
    ) -> torch.FloatTensor:
    assert rewards.ndim == 2
    assert rewards.size() == values.size() == ends.size()
    assert value_bootstrap.ndim == 1 and value_bootstrap.size(0) == rewards.size(0)

    lambda_returns = torch.empty_like(values)
    lambda_returns = rewards + ends.logical_not() * gamma * (1 - lambda_) * torch.cat((values[:, 1:], value_bootstrap.unsqueeze(1)), dim=1)

    last = value_bootstrap
    for t in list(range(rewards.size(1)))[::-1]:
        lambda_returns[:, t] += ends[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns
