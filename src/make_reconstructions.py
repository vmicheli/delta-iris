import shutil 

from einops import rearrange
import numpy as np
from PIL import Image
import torch


@torch.no_grad()
def make_reconstructions_from_batch(batch, save_dir, epoch, tokenizer, should_overwrite):

    if should_overwrite and save_dir.is_dir():
        shutil.rmtree(save_dir)

    save_dir.mkdir(exist_ok=True, parents=False)

    original_frames = tensor_to_np_frames(rearrange(batch.observations, 'b t c h w -> b t h w c'))
    rec_frames = generate_reconstructions_with_tokenizer(batch, tokenizer)
    diff_frames = 255 - np.abs(rec_frames.astype(float) - original_frames.astype(float)).mean(axis=-1, keepdims=True).astype(np.uint8).repeat(3, axis=-1)

    for i, image in enumerate(map(Image.fromarray, np.concatenate(list(np.concatenate((original_frames, rec_frames, diff_frames), axis=-3)), axis=-2))):
        image.save(save_dir / f'epoch_{epoch:04d}_t_{i:04d}.png')


def tensor_to_np_frames(inputs):
    check_float_btw_0_1(inputs)
    return inputs.mul(255).cpu().numpy().astype(np.uint8)


def check_float_btw_0_1(inputs):
    assert inputs.is_floating_point() and (inputs >= 0).all() and (inputs <= 1).all()


@torch.no_grad()
def generate_reconstructions_with_tokenizer(batch, tokenizer):
    r = reconstruct_through_tokenizer(batch.observations, batch.actions, tokenizer)
    rec_frames = tensor_to_np_frames(rearrange(r, 'b t c h w -> b t h w c'))
    return rec_frames


@torch.no_grad()
def reconstruct_through_tokenizer(obs, act, tokenizer):
    check_float_btw_0_1(obs)
    x1 = obs[:, :-1]
    a = act[:, :-1]
    x2 = obs[:, 1:]
    r = tokenizer.encode_decode(x1, a, x2)
    r = torch.cat((obs[:, 0:1], r), dim=1)
    return r
