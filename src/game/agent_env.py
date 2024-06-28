from einops import rearrange
import numpy as np
from PIL import Image
import torch

from agent import Agent
from envs import SingleProcessEnv, WorldModelEnv
from game.keymap import get_keymap_and_action_names


class AgentEnv:
    def __init__(self, agent: Agent, env: SingleProcessEnv, keymap_name: str) -> None:
        assert isinstance(env, SingleProcessEnv) or isinstance(env, WorldModelEnv)

        self.agent = agent
        self.env = env
        _, self.action_names = get_keymap_and_action_names(keymap_name)

        self.obs = None
        self._t = None
        self._return = None

    def _to_tensor(self, obs: np.ndarray) -> torch.FloatTensor:
        return rearrange(torch.FloatTensor(obs).div(255), '... h w c -> ... c h w').to(self.agent.device)

    def _to_array(self, obs: torch.FloatTensor) -> np.ndarray:
        return rearrange(obs[0].mul(255), '... c h w -> ... h w c').cpu().numpy().astype(np.uint8)

    def reset(self) -> torch.FloatTensor:
        obs = self.env.reset()
        self.obs = self._to_tensor(obs) if isinstance(self.env, SingleProcessEnv) else obs[1].frames[:, 0]
        self.agent.actor_critic.reset(n=1)
        self._t = 0
        self._return = 0

        return obs

    def step(self, *args, **kwargs) -> torch.FloatTensor:
        with torch.no_grad():
            act, value = self.agent.actor_critic.act(self.obs, should_sample=True)

        obs, reward, done, _ = self.env.step(act.cpu().numpy())
        self.obs = self._to_tensor(obs) if isinstance(self.env, SingleProcessEnv) else obs.frames[:, 0]
        self._t += 1
        self._return += reward[0]

        info = {
            'timestep': self._t,
            'action': self.action_names[act[0]],
            'return': round(self._return, 2),
            'value': round(value.item(), 2),
        }

        return obs, reward, done, info

    def render(self) -> Image.Image:
        arr = self._to_array(self.obs)

        return Image.fromarray(arr)


