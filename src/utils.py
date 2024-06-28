from collections import OrderedDict
import cv2
from pathlib import Path
import random
import shutil
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from data import Episode


def configure_optimizer(model: nn.Module, learning_rate: float, weight_decay: float, *blacklist_module_names) -> AdamW:
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, nn.Conv2d, nn.GroupNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=learning_rate)

    return optimizer


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict: Dict, module_name: str) -> OrderedDict:
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


@torch.no_grad()
def compute_discounted_returns(rewards: torch.FloatTensor, gamma: float) -> torch.FloatTensor:
    assert 0 < gamma <= 1 and rewards.ndim == 2  # (B, T)
    gammas = gamma ** torch.arange(rewards.size(1))
    r = rewards * gammas

    return (r + r.sum(dim=1, keepdim=True) - r.cumsum(dim=1)) / gammas


class LossWithIntermediateLosses:
    def __init__(self, **kwargs) -> None:
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}


class EpisodeDirManager:
    def __init__(self, episode_dir: Path, max_num_episodes: int) -> None:
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=False, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf')

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        ep_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith('episode_')]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(ep_paths, key=lambda ep_path: int(ep_path.stem.split('_')[1]))
            to_remove.unlink()
        torch.save(episode.__dict__, self.episode_dir / f'episode_{episode_id}_epoch_{epoch}.pt')

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            self.best_return = ep_return
            path_best_ep = [p for p in self.episode_dir.iterdir() if p.stem.startswith('best_')]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            torch.save(episode.__dict__, self.episode_dir / f'best_episode_{episode_id}_epoch_{epoch}.pt')


class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)

        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames):
    assert frames.ndim == 4  # (T, H, W, C)
    _, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()


def try_until_no_except(fn: Callable):
    while True:
        try:
            fn()
        except:
            continue
        else:
            break


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x: torch.FloatTensor, x_min: int = -20, x_max: int = 20, num_buckets: int = 255) -> torch.FloatTensor:
    x.clamp_(x_min, x_max - 1e-5)
    buckets = torch.linspace(x_min, x_max, num_buckets).to(x.device)
    k = torch.searchsorted(buckets, x) - 1
    values = torch.stack((buckets[k + 1] - x, x - buckets[k]), dim=-1) / (buckets[k + 1] - buckets[k]).unsqueeze(-1)  
    two_hots = torch.scatter(x.new_zeros(*x.size(), num_buckets), dim=-1, index=torch.stack((k, k + 1), dim=-1), src=values)

    return two_hots


def compute_softmax_over_buckets(logits: torch.FloatTensor, x_min: int = -20, x_max: int = 20, num_buckets: int = 255) -> torch.FloatTensor:
    buckets = torch.linspace(x_min, x_max, num_buckets).to(logits.device)
    probs = F.softmax(logits, dim=-1)

    return probs @ buckets


def plot_counts(counts: np.ndarray) -> Image:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(counts)
    p = Path('priorities.png')
    fig.savefig(p)
    plt.close(fig)
    im = Image.open(p)
    p.unlink()

    return im


def compute_mask_after_first_done(ends: torch.LongTensor) -> torch.BoolTensor:
    assert ends.ndim == 2
    first_one_index = torch.argmax(ends, dim=1)
    mask = torch.arange(ends.size(1), device=ends.device).unsqueeze(0) <= first_one_index.unsqueeze(1)
    mask = torch.logical_or(mask, ends.sum(dim=1, keepdim=True) == 0)

    return mask
