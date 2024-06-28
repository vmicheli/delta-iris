from pathlib import Path

import torch
import torch.nn as nn

from models import ActorCritic, Tokenizer, WorldModel
from utils import extract_state_dict


class Agent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

        print(f'{sum(p.numel() for p in tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in world_model.parameters())} parameters in agent.world_model')
        print(f'{sum(p.numel() for p in actor_critic.parameters())} parameters in agent.actor_critic')

    @property
    def device(self) -> torch.device:
        return self.actor_critic.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True, strict: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)

        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'), strict=strict)
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'), strict=strict)
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'), strict=strict)
