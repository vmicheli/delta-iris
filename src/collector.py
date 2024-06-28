import random
import sys
from typing import Dict, List, Optional, Union

from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
import wandb

from agent import Agent
from data import collate_segments_to_batch, Episode, EpisodeCountManager, EpisodeDataset, make_segment, SegmentId
from envs import SingleProcessEnv, MultiProcessEnv
from utils import EpisodeDirManager, RandomHeuristic


class Collector:
    def __init__(
            self,
            env: Union[SingleProcessEnv, MultiProcessEnv],
            dataset: EpisodeDataset,
            episode_dir_manager: EpisodeDirManager,
            episode_count_manager: Optional[EpisodeCountManager] = None
        ) -> None:
  
        self.env = env
        self.dataset = dataset
        self.episode_dir_manager = episode_dir_manager
        self.episode_count_manager = episode_count_manager
        self.episode_ids = [None] * self.env.num_envs
        self.heuristic = RandomHeuristic(self.env.num_actions)

        self.obs = None
        self.excess_steps = 0  # number of steps to deduct from next collect if collecting more than expected with multi-process env (at most num_envs - 1)

    @torch.no_grad()
    def collect(
        self,
        agent: Agent,
        epoch: int,
        epsilon: float,
        should_sample: bool,
        temperature: float,
        burn_in_length: int,
        *,
        num_steps_first_epoch: Optional[int] = None,
        num_steps: Optional[int] = None,
        num_episodes: Optional[int] = None,
        offset_episode_id: int = 0
        ) -> List[Dict]:

        assert self.env.num_actions == agent.world_model.config.num_actions
        assert 0 <= epsilon <= 1
        assert (num_steps is None) != (num_episodes is None)

        if epoch == 1 and num_steps_first_epoch is not None:
            epsilon = 1
            num_steps = num_steps_first_epoch

        if num_steps is not None:
            num_steps -= self.excess_steps
            num_steps = max(0, num_steps)
            self.excess_steps = 0

        should_stop = lambda steps, episodes: steps >= num_steps if num_steps is not None else episodes >= num_episodes

        agent.eval()
        agent.actor_critic.reset(n=self.env.num_envs)
        if self.obs is None:
            self.obs = self.env.reset()

        to_log = []
        steps, episodes = 0, 0
        returns = []
        observations, actions, rewards, dones = [], [], [], []

        if burn_in_length > 0 and epsilon < 1:
            if set(self.episode_ids) == {None}:
                h, w, c = self.obs.shape[1:]
                burn_in_obs = torch.zeros(self.obs.shape[0], burn_in_length, c, h, w, device=agent.device)
            else:
                segment_ids = [SegmentId(episode_id, start=self.dataset.lengths[episode_id] - burn_in_length, stop=self.dataset.lengths[episode_id]) for episode_id in self.episode_ids]
                segments = [make_segment(self.dataset.load_episode(episode_id), segment_id, should_pad=True) for episode_id, segment_id in zip(self.episode_ids, segment_ids)]
                burn_in_batch = collate_segments_to_batch(segments).to(agent.device)
                burn_in_obs = burn_in_batch.observations

            for t in range(burn_in_length):
                _ = agent.actor_critic.act(burn_in_obs[:, t])

        pbar = tqdm(total=num_steps if num_steps is not None else num_episodes, desc=f'Experience collection ({self.dataset.name})', file=sys.stdout)

        while not should_stop(steps, episodes):
            observations.append(self.obs)
            obs = rearrange(torch.FloatTensor(self.obs).div(255), 'n h w c -> n c h w').to(agent.device)
            
            if epsilon < 1:
                act, _ = agent.actor_critic.act(obs, should_sample=should_sample, temperature=temperature)

            if random.random() < epsilon:
                act = self.heuristic.act(obs)

            act = act.cpu().numpy()
            self.obs, reward, done, _ = self.env.step(act)

            actions.append(act)
            rewards.append(reward)
            dones.append(done)

            new_steps = len(self.env.mask_new_dones)
            steps += new_steps
            pbar.update(new_steps if num_steps is not None else 0)

            if self.env.should_reset():             
                self.add_experience_to_dataset(observations, actions, rewards, dones)

                episodes += self.env.num_envs
                pbar.update(self.env.num_envs if num_episodes is not None else 0)

                for episode_id in self.episode_ids:
                    episode = self.dataset.load_episode(episode_id)
                    self.episode_dir_manager.save(episode, offset_episode_id + episode_id, epoch)
                    metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                    returns.append(metrics_episode['episode_return'])
                    metrics_episode['episode_num'] = offset_episode_id + episode_id
                    metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np.histogram(episode.actions.numpy(), bins=np.arange(0, self.env.num_actions + 1) - 0.5, density=True))
                    to_log.append({f'{self.dataset.name}/{k}': v for k, v in metrics_episode.items()})

                self.obs = self.env.reset()
                self.episode_ids = [None] * self.env.num_envs
                agent.actor_critic.reset(n=self.env.num_envs)
                observations, actions, rewards, dones = [], [], [], []

        if len(observations) > 0:
            self.add_experience_to_dataset(observations, actions, rewards, dones)

        if num_steps is not None:
            self.excess_steps = steps - num_steps

        agent.actor_critic.clear()

        metrics_collect = {'#episodes': self.dataset.num_episodes, '#steps': self.dataset.num_steps}
        if len(returns) > 0:
            metrics_collect['return'] = np.mean(returns)
        metrics_collect = {f'{self.dataset.name}/{k}': v for k, v in metrics_collect.items()}
        to_log.append(metrics_collect)

        return to_log

    def add_experience_to_dataset(self, observations: List[np.ndarray], actions: List[np.ndarray], rewards: List[np.ndarray], dones: List[np.ndarray]) -> None:
        for i, (o, a, r, d) in enumerate(zip(*map(lambda arr: np.swapaxes(arr, 0, 1), [observations, actions, rewards, dones]))):  # (T, N, ...) -> (N, T, ...)
            episode = Episode(
                observations=torch.ByteTensor(o).permute(0, 3, 1, 2).contiguous(),  # channel-first
                actions=torch.LongTensor(a),
                rewards=torch.FloatTensor(r),
                ends=torch.LongTensor(d)
                )
            self.episode_ids[i] = self.dataset.add_episode(episode, episode_id=self.episode_ids[i])
            
            if self.episode_count_manager is not None:
                self.episode_count_manager.add_episode(episode_id=self.episode_ids[i])
