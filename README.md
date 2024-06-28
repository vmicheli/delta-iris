# Efficient World Models with Context-Aware Tokenization (Δ-IRIS)

[Efficient World Models with Context-Aware Tokenization](https://openreview.net/forum?id=BiWIERWBFX) <br>
[Vincent Micheli](https://vmicheli.github.io)\*, [Eloi Alonso](https://eloialonso.github.io)\*, [François Fleuret](https://fleuret.org/francois/) <br>

**TL;DR** Δ-IRIS is a reinforcement learning agent trained in the imagination of its world model.

<div align='center'>
Δ-IRIS agent alternatively playing in the environment and its world model

https://github.com/vmicheli/delta-iris/assets/32040353/ff2dc7a7-fa0a-4338-8f77-1637dff8642d


</div>





## Setup

- `pip install pip==23.0`
- Install [dependencies](requirements.txt): `pip install -r requirements.txt`
- Warning: Atari ROMs will be downloaded with the Atari dependencies, which means that you acknowledge that you have the license to use them.

## Launch a training run

**Crafter:**
```bash
python src/main.py
```

The run will be located in `outputs/YYYY-MM-DD/hh-mm-ss/`.

By default, logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn logging off.

**Atari:**
```bash
python src/main.py env=atari params=atari env.train.id=BreakoutNoFrameskip-v4
```

Note that this Atari configuration achieves slightly higher aggregate metrics than those reported in the paper. Here is the [updated table of results](https://github.com/user-attachments/files/16022861/results_atari_updated.pdf).


## Configuration

- All configuration files are located in `config/`, the main configuration file is `config/trainer.yaml`.
- The simplest way to customize the configuration is to edit these files directly.
- Please refer to [Hydra](https://github.com/facebookresearch/hydra) for more details regarding configuration management.

## Run folder

Each new run is located in `outputs/YYYY-MM-DD/hh-mm-ss/`. This folder is structured as:

```txt
outputs/YYYY-MM-DD/hh-mm-ss/
│
└─── checkpoints
│   │   last.pt
│   │   optimizer.pt
│   │   ...
│   │
│   └── dataset
│      │
│      └─ train
│        │   info.pt
│        │   ...
│      │
│      └─ test
│        │   info.pt
│        │   ...
│
└─── config
│   │   trainer.yaml
│   │   ...
│
└─── media
│   │
│   └── episodes
│      │   ...
│   │
│   └── reconstructions
│      │   ...
│
└─── scripts
│   │   resume.sh
│   │   play.sh
│
└─── src
│   │   main.py
│   │   ...
│
└─── wandb
    │   ...
```

- `checkpoints`: contains the last checkpoint of the model, its optimizer and the dataset.
- `media`:
  - `episodes`: contains train / test episodes for visualization purposes.
  - `reconstructions`: contains original frames alongside their reconstructions with the autoencoder.
- `scripts`: **from the run folder**, you can use the following scripts.
  - `resume.sh`: Launch `./scripts/resume.sh` to resume a training run that crashed.
  - `play.sh`: Tool to visualize the agent and interact with the world model.
    - Launch `./scripts/play.sh` to watch the agent play live in the environment.
    - Launch `./scripts/play.sh -w` to play live in the world model. Note that for faster interaction, the memory of the world model is flushed after a few seconds.
    - Launch `./scripts/play.sh -a` to watch the agent play live in the world model. Note that for faster interaction, the memory of the world model is flushed after a few seconds.
    - Launch `./scripts/play.sh -e` to visualize the episodes contained in `media/episodes`.
    - Add the flag `-h` to display a header with additional information.

## Pretrained agent

An agent checkpoint (Crafter 5M frames) is available [here](https://drive.google.com/file/d/16qBdJA2OvKd-5OgS9IX8Qxkos9QhhK4_/view?usp=sharing).

To visualize the agent or play in its world model:
- Create a `checkpoints` directory
- Copy the checkpoint to `checkpoints/last.pt`
- Run `./scripts/play.sh` with the flags of your choice as described above.

## Cite

If you find this code or paper useful, please use the following reference:

```
@inproceedings{
micheli2024efficient,
title={Efficient World Models with Context-Aware Tokenization},
author={Vincent Micheli and Eloi Alonso and François Fleuret},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=BiWIERWBFX}
}
```

## Credits

- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- [https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)
- You might also want to check out our [codebase](https://github.com/eloialonso/iris) for [IRIS](https://arxiv.org/abs/2209.00588)
