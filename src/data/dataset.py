from pathlib import Path
import shutil
from typing import Dict, Optional, Union

import numpy as np
import torch

from .episode import Episode
from .segment import Segment, SegmentId
from .utils import make_segment


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, directory: Path, name: str) -> None:
        super().__init__()
        self.name = name
        self.directory = Path(directory)
        self.num_episodes, self.num_steps, self.start_idx, self.lengths = None, None, None, None

        if not self.directory.is_dir():
            self._init_empty()
        else:
            self._load_info()
            print(f'({name}) {self.num_episodes} episodes, {self.num_steps} steps.')

    @property
    def info_path(self) -> Path:
        return self.directory / 'info.pt'

    @property
    def info(self) -> Dict[str, Union[int, np.ndarray]]:
        return {'num_episodes': self.num_episodes, 'num_steps': self.num_steps, 'start_idx': self.start_idx, 'lengths': self.lengths}

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        return self._load_segment(segment_id)

    def _init_empty(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=False)
        self.num_episodes = 0
        self.num_steps = 0
        self.start_idx = np.array([], dtype=np.int64)            
        self.lengths = np.array([], dtype=np.int64)
        self.save_info()

    def _load_info(self) -> None:
        info = torch.load(self.info_path)
        self.num_steps = info['num_steps']
        self.num_episodes = info['num_episodes']
        self.start_idx = info['start_idx']
        self.lengths = info['lengths']

    def save_info(self) -> None:
        torch.save(self.info, self.info_path)

    def clear(self) -> None:
        shutil.rmtree(self.directory)
        self._init_empty()

    def _get_episode_path(self, episode_id: int) -> Path:
        n = 3 # number of hierarchies
        powers = np.arange(n)
        subfolders = list(map(int, np.floor((episode_id % 10 ** (1 + powers)) / 10 ** powers) * 10 ** powers))[::-1]
        return self.directory / '/'.join(list(map(lambda x: f'{x[1]:0{n - x[0]}d}', enumerate(subfolders)))) / f'{episode_id}.pt'

    def _load_segment(self, segment_id: SegmentId, should_pad: bool = True) -> Segment:
        episode = self.load_episode(segment_id.episode_id)
        return make_segment(episode, segment_id, should_pad)

    def load_episode(self, episode_id: int) -> Episode:
        return Episode(**torch.load(self._get_episode_path(episode_id)))

    def add_episode(self, episode: Episode, *, episode_id: Optional[int] = None) -> int:
        if episode_id is None:
            episode_id = self.num_episodes
            self.start_idx = np.concatenate((self.start_idx, np.array([self.num_steps])))
            self.lengths = np.concatenate((self.lengths, np.array([len(episode)])))
            self.num_steps += len(episode)
            self.num_episodes += 1

        else:
            assert episode_id < self.num_episodes
            old_episode = self.load_episode(episode_id) 
            episode = old_episode.merge(episode)
            incr_num_steps = len(episode) - len(old_episode)
            self.lengths[episode_id] = len(episode)
            self.start_idx[episode_id + 1:] += incr_num_steps            
            self.num_steps += incr_num_steps

        episode_path = self._get_episode_path(episode_id)
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(episode.__dict__, episode_path.with_suffix('.tmp'))
        episode_path.with_suffix('.tmp').rename(episode_path)

        return episode_id

    def get_episode_id_from_global_idx(self, global_idx: np.ndarray) -> np.ndarray:
        return (np.argmax(self.start_idx.reshape(-1, 1) > global_idx, axis=0) - 1) % self.num_episodes

    def get_global_idx_from_segment_id(self, segment_id: SegmentId) -> np.ndarray:
        start_idx = self.start_idx[segment_id.episode_id]
        return np.arange(start_idx + segment_id.start, start_idx + segment_id.stop)
