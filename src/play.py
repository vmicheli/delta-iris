from functools import partial 
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from agent import Agent
from envs import SingleProcessEnv, WorldModelEnv
from game import AgentEnv, EpisodeReplayEnv, Game
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from models.actor_critic import ActorCritic

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    assert cfg.mode in ('episode_replay', 'agent_in_env', 'agent_in_world_model', 'play_in_world_model')

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    if cfg.mode.startswith('agent_in_'):
        h, w, _ = test_env.env.unwrapped.observation_space.shape
    else:
        h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]

    if cfg.mode == 'episode_replay':
        env = EpisodeReplayEnv(replay_keymap_name=cfg.env.keymap, episode_dir=Path('media/episodes'), agent=None)
        keymap = 'episode_replay'

    else:
        cfg.params.tokenizer.num_actions = cfg.params.world_model.num_actions = cfg.params.actor_critic.model.num_actions = test_env.num_actions
        agent = Agent(Tokenizer(instantiate(cfg.params.tokenizer)), WorldModel(instantiate(cfg.params.world_model)), ActorCritic(instantiate(cfg.params.actor_critic))).to(device)
        agent.load(Path('checkpoints/last.pt'), device=device, strict=False)        
        agent.eval()

        if cfg.mode == 'play_in_world_model':
            env = WorldModelEnv(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device, env=env_fn())
            keymap = cfg.env.keymap
            
        elif cfg.mode == 'agent_in_env':
            env = AgentEnv(agent, test_env, cfg.env.keymap)
            keymap = 'empty'

        elif cfg.mode == 'agent_in_world_model':
            wm_env = WorldModelEnv(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device, env=env_fn())
            env = AgentEnv(agent, wm_env, cfg.env.keymap)
            keymap = 'empty'

        else:
            raise NotImplementedError
    
    game = Game(env, keymap_name=keymap, size=size, fps=cfg.fps, verbose=bool(cfg.header), record_mode=bool(cfg.save_mode))
    game.run()


if __name__ == "__main__":
    main()
