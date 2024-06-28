from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass
class EpisodeMetrics:
    episode_length: int
    episode_return: float


@dataclass
class Episode:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    ends: torch.LongTensor

    def __post_init__(self):
        assert len(self.observations) == len(self.actions) == len(self.rewards) == len(self.ends)
        if self.ends.sum() > 0:
            idx_end = torch.argmax(self.ends) + 1
            self.observations = self.observations[:idx_end]
            self.actions = self.actions[:idx_end]
            self.rewards = self.rewards[:idx_end]
            self.ends = self.ends[:idx_end]

    def __len__(self) -> int:
        return self.observations.size(0)

    def merge(self, other: Episode) -> Episode:
        return Episode(
            torch.cat((self.observations, other.observations), dim=0),
            torch.cat((self.actions, other.actions), dim=0),
            torch.cat((self.rewards, other.rewards), dim=0),
            torch.cat((self.ends, other.ends), dim=0),
        )

    def compute_metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(len(self), self.rewards.sum())
