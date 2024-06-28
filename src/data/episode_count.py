from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .dataset import EpisodeDataset


class EpisodeCountManager:
    def __init__(self, dataset: EpisodeDataset) -> None:
        self.dataset = dataset
        self.all_counts = dict()

    def load(self, path_to_checkpoint: Path) -> None:
        self.all_counts = torch.load(path_to_checkpoint)
        assert all([counts.shape[0] == self.dataset.num_episodes for counts in self.all_counts.values()])

    def save(self, path_to_checkpoint: Path) -> None:
        torch.save(self.all_counts, path_to_checkpoint)

    def register(self, *keys: Tuple[str]) -> None:
        assert all([key not in self.all_counts for key in keys])
        self.all_counts.update({key: np.zeros(self.dataset.num_episodes, dtype=np.int64) for key in keys}) 

    def add_episode(self, episode_id: int) -> None:
        for key, counts in self.all_counts.items():
            assert episode_id <= counts.shape[0]
            if episode_id == counts.shape[0]:
                self.all_counts[key] = np.concatenate((counts, np.zeros(1, dtype=np.int64)))
            assert self.all_counts[key].shape[0] == self.dataset.num_episodes

    def increment_episode_count(self, key: str, episode_id: int) -> None:
        assert key in self.all_counts
        self.all_counts[key][episode_id] += 1

    def compute_probabilities(self, key: str, alpha: float) -> np.ndarray:
        assert key in self.all_counts
        inverse_counts = 1 / (1 + self.all_counts[key])
        p = inverse_counts ** alpha
        return p / p.sum()
