from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

from envs.world_model_env import WorldModelEnvOutput
from .utils import ActorCriticOutput
from ..convnet import FrameCnnConfig, FrameEncoder


class CnnLstmActorCritic(nn.Module):
    def __init__(self, frame_encoder_config: FrameCnnConfig, num_actions: int, two_hot_rets: bool) -> None:
        super().__init__()
        self.lstm_dim = 512
        self.hx, self.cx = None, None
        self.cnn = nn.Sequential(FrameEncoder(frame_encoder_config), Rearrange('b t c h w -> (b t) (h w c)'))
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)

        self.actor_linear = nn.Linear(self.lstm_dim, num_actions)
        self.critic_linear = nn.Linear(self.lstm_dim, 255 if two_hot_rets else 1)
        self.critic_linear.weight.data.zero_()
        self.critic_linear.bias.data.zero_()

    def forward(self, wm_env_output: WorldModelEnvOutput, **kwargs) -> ActorCriticOutput:
        obs = wm_env_output.frames
        assert obs.ndim == 5 and obs.size(0) == self.hx.size(0)
 
        for i in range(obs.size(1)):
            inputs = obs[:, i:i+1]
            x = self.cnn(inputs)
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))

        logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
        logits_values = rearrange(self.critic_linear(self.hx), 'b c -> b 1 c')

        return ActorCriticOutput(logits_actions, logits_values)

    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh.device

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int) -> None:
        self.hx = torch.zeros(n, self.lstm_dim, device=self.device)
        self.cx = torch.zeros(n, self.lstm_dim, device=self.device)

    @torch.no_grad()
    def burn_in(self, wm_env_output: WorldModelEnvOutput) -> None:
        _ = self(wm_env_output)

    def build_present_input_from_past(self, past_obs) -> torch.LongTensor:
        return WorldModelEnvOutput(past_obs[:, -1:], None)
