from pathlib import Path

import numpy as np
from PIL import Image
import torch

from agent import Agent
from data import Episode
from .keymap import get_keymap_and_action_names


class EpisodeReplayEnv:
    def __init__(self, replay_keymap_name: str, episode_dir: Path, agent: Agent) -> None:
        _, self.action_names = get_keymap_and_action_names(replay_keymap_name)
        print(episode_dir)
        assert episode_dir.is_dir()
        self._paths = {}
        for mode in ['train', 'test']:
            directory = episode_dir / mode
            if directory.is_dir():
                self._paths[mode] = sorted([p for p in directory.iterdir() if p.suffix == '.pt'])
                print(f'Found {len(self._paths[mode])} {mode} episodes.')
            else:
                print(f'No {mode} episodes.')

        self.agent = agent
        self._t, self._episode = None, None
        self._ep_idx = 0
        self._mode = 'train'
        self.load()

    def load(self) -> None:
        self._episode = Episode(**torch.load(self.paths[self._ep_idx]))
        self._t = 0

    def load_next(self) -> None:
        self._ep_idx = (self._ep_idx + 1) % len(self.paths)
        self.load()

    def load_previous(self) -> None:
        self._ep_idx = (self._ep_idx - 1) % len(self.paths)
        self.load()

    def set_mode(self, mode: str) -> None:
        assert mode in ['train', 'test']
        if mode in self._paths:
            self._mode = mode
            self._ep_idx = 0
            self.load()
        else:
            print(f'No {mode} episodes.')

    def __len__(self) -> int:
        return len(self.ends)

    @property
    def paths(self):
        return self._paths[self._mode]

    @property
    def values(self):
        return self._values
    
    @property
    def observations(self):
        return self._episode.observations

    @property
    def actions(self):
        return self._episode.actions

    @property
    def rewards(self):
        return self._episode.rewards

    @property
    def ends(self):
        return self._episode.ends

    def reset(self) -> torch.FloatTensor:
        return self.observations[self._t]

    def step(self, action) -> torch.FloatTensor:
        if action == 1:
            self._t = (self._t - 1) % len(self)
        elif action == 2:
            self._t = (self._t + 1) % len(self)
        if action == 3:
            self._t = (self._t - 10) % len(self)
        elif action == 4:
            self._t = (self._t + 10) % len(self)
        elif action == 5:
            self._t = 0
        elif action == 6:
            self.load_previous()
        elif action == 7:
            self.load_next()
        elif action == 8:
            self.set_mode('train')
        elif action == 9:
            self.set_mode('test')

        act = self.actions[self._t]
        reward = self.rewards[self._t].item()
        done = self.ends[self._t].item()
        info = {
            'ep_name': f'[{self._mode}] {self.paths[self._ep_idx].stem}',
            'timestep': self._t,
            'action': self.action_names[act],
            'cum_reward': f'{sum(self.rewards[:self._t + 1]):.3f}',
        }

        return self.observations[self._t], reward, done, info

    def render(self) -> Image.Image:
        obs = self.observations[self._t]
        arr = obs.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(arr)
