import gym
import pygame


def get_keymap_and_action_names(name):

    if name == 'empty':
        return EMPTY_KEYMAP, EMPTY_ACTION_NAMES

    if name == 'episode_replay':
        return EPISODE_REPLAY_KEYMAP, EPISODE_REPLAY_ACTION_NAMES

    if name == 'crafter':
        return CRAFTER_KEYMAP, CRAFTER_ACTION_NAMES

    if name == 'atari':
        return ATARI_KEYMAP, ATARI_ACTION_NAMES

    assert name.startswith('atari/')
    env_id = name.split('atari/')[1]
    action_names = [x.lower() for x in gym.make(env_id).get_action_meanings()]
    keymap = {}
    for key, value in ATARI_KEYMAP.items():
        if ATARI_ACTION_NAMES[value] in action_names:
            keymap[key] = action_names.index(ATARI_ACTION_NAMES[value])
    return keymap, action_names


ATARI_ACTION_NAMES = [
    'noop',
    'fire',
    'up',
    'right',
    'left',
    'down',
    'upright',
    'upleft',
    'downright',
    'downleft',
    'upfire',
    'rightfire',
    'leftfire',
    'downfire',
    'uprightfire',
    'upleftfire',
    'downrightfire',
    'downleftfire',
]

ATARI_KEYMAP = {
    pygame.K_SPACE: 1,

    pygame.K_w: 2,
    pygame.K_d: 3,
    pygame.K_a: 4,
    pygame.K_s: 5,

    pygame.K_t: 6,
    pygame.K_r: 7,
    pygame.K_g: 8,
    pygame.K_f: 9,

    pygame.K_UP: 10,
    pygame.K_RIGHT: 11,
    pygame.K_LEFT: 12,
    pygame.K_DOWN: 13,

    pygame.K_u: 14,
    pygame.K_y: 15,
    pygame.K_j: 16,
    pygame.K_h: 17,
}

CRAFTER_ACTION_NAMES = [
    'noop',
    'move_left',
    'move_right', 
    'move_up',
    'move_down',
    'do',
    'sleep',
    'place_stone',
    'place_table',
    'place_furnace',
    'place_plant',
    'make_wood_pickaxe',
    'make_stone_pickaxe',
    'make_iron_pickaxe',
    'make_wood_sword',
    'make_stone_sword',
    'make_iron_sword'
]

CRAFTER_KEYMAP = {
    pygame.K_a: 1,
    pygame.K_d: 2,
    pygame.K_w: 3,
    pygame.K_s: 4,
    pygame.K_SPACE: 5,
    pygame.K_TAB: 6,

    pygame.K_r: 7,
    pygame.K_t: 8,
    pygame.K_f: 9,
    pygame.K_p: 10,

    pygame.K_1: 11,
    pygame.K_2: 12,
    pygame.K_3: 13,
    pygame.K_4: 14,
    pygame.K_5: 15,
    pygame.K_6: 16,
}

EPISODE_REPLAY_ACTION_NAMES = [
    'noop',
    'previous',
    'next',
    'previous_10',
    'next_10',
    'go_to_start',
    'load_previous',
    'load_next',
    'go_to_train_episodes',
    'go_to_test_episodes',
]

EPISODE_REPLAY_KEYMAP = {
    pygame.K_LEFT: 1,
    pygame.K_RIGHT: 2,
    pygame.K_PAGEDOWN: 3,
    pygame.K_PAGEUP: 4,
    pygame.K_SPACE: 5,
    pygame.K_DOWN: 6,
    pygame.K_UP: 7,
    pygame.K_t: 8,
    pygame.K_y: 9,
}

EMPTY_ACTION_NAMES = [
    'noop',
]

EMPTY_KEYMAP = {
}