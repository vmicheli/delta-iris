import math
from typing import Generator, List

import torch

from .batch import Batch
from .episode import Episode
from .segment import Segment, SegmentId


def collate_segments_to_batch(segments: List[Segment]) -> Batch:
    return Batch(
        torch.stack(list(map(lambda s: s.observations, segments))).div(255),
        torch.stack(list(map(lambda s: s.actions, segments))),
        torch.stack(list(map(lambda s: s.rewards, segments))),
        torch.stack(list(map(lambda s: s.ends, segments))),
        torch.stack(list(map(lambda s: s.mask_padding, segments))),
        list(map(lambda segment: segment.id, segments))
    )


def make_segment(episode: Episode, segment_id: SegmentId, should_pad: bool = True) -> Segment:
    assert segment_id.start < len(episode) and segment_id.stop > 0 and segment_id.start < segment_id.stop
    padding_length_right = max(0, segment_id.stop - len(episode))
    padding_length_left = max(0, -segment_id.start)
    assert padding_length_right == padding_length_left == 0 or should_pad

    def pad(x):
        pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
        return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

    start = max(0, segment_id.start)
    stop = min(len(episode), segment_id.stop)

    return Segment(
        pad(episode.observations[start:stop]),
        pad(episode.actions[start:stop]),
        pad(episode.rewards[start:stop]),
        pad(episode.ends[start:stop]),
        mask_padding=torch.cat((torch.zeros(padding_length_left), torch.ones(stop - start), torch.zeros(padding_length_right))).bool(),
        id=SegmentId(segment_id.episode_id, start, stop)
    )


class DatasetTraverser:
    def __init__(self, dataset, batch_num_samples: int, chunk_size: int) -> None:
        self.dataset = dataset
        self.batch_num_samples = batch_num_samples
        self.chunk_size = chunk_size
        self._num_batches = math.ceil(sum([math.ceil(dataset.lengths[episode_id] / chunk_size) - int(dataset.lengths[episode_id] % chunk_size == 1) for episode_id in range(dataset.num_episodes)]) / batch_num_samples)

    def __len__(self) -> int:
        return self._num_batches 

    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []

        for episode_id in range(self.dataset.num_episodes):
            episode = self.dataset.load_episode(episode_id)
            chunks.extend(make_segment(episode, SegmentId(episode_id, start=i * self.chunk_size, stop=(i + 1) * self.chunk_size), should_pad=True) for i in range(math.ceil(len(episode) / self.chunk_size)))
            if chunks[-1].effective_size < 2:
                chunks.pop()

            while len(chunks) >= self.batch_num_samples:
                yield collate_segments_to_batch(chunks[:self.batch_num_samples])
                chunks = chunks[self.batch_num_samples:]

        if len(chunks) > 0:
            yield collate_segments_to_batch(chunks)
