from typing import Generator, List

import numpy as np
import torch

from .dataset import EpisodeDataset
from .segment import SegmentId


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: EpisodeDataset, num_steps_per_epoch: int, batch_size: int, sequence_length: int, can_sample_beyond_end: bool) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.probabilities = None
        self.num_steps_per_epoch = num_steps_per_epoch
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.can_sample_beyond_end = can_sample_beyond_end

    def __len__(self) -> int:
        return self.num_steps_per_epoch

    def __iter__(self) -> Generator[List[SegmentId], None, None]:
        for _ in range(self.num_steps_per_epoch):
            yield self.sample()

    def sample(self) -> List[SegmentId]:
        episode_ids = np.random.choice(np.arange(self.dataset.num_episodes), size=self.batch_size, replace=True, p=self.probabilities)
        timesteps = np.random.randint(low=0, high=self.dataset.lengths[episode_ids])

        # padding allowed, both before start and after end
        if self.can_sample_beyond_end:  
            starts = timesteps - np.random.randint(0, self.sequence_length, len(timesteps))
            stops = starts + self.sequence_length

        # padding allowed only before start
        else:                           
            stops = np.minimum(self.dataset.lengths[episode_ids], timesteps + 1 + np.random.randint(0, self.sequence_length, len(timesteps)))
            starts = stops - self.sequence_length

        return list(map(lambda x: SegmentId(*x), zip(episode_ids, starts, stops)))

