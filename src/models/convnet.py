from dataclasses import dataclass
from typing import List

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FrameCnnConfig:
    image_channels: int
    latent_dim: int
    num_channels: int
    mult: List[int]
    down: List[int]


class FrameEncoder(nn.Module):
    def __init__(self, config: FrameCnnConfig) -> None:
        super().__init__()

        assert len(config.mult) == len(config.down)
        encoder_layers = [nn.Conv2d(config.image_channels, config.num_channels, kernel_size=3, stride=1, padding=1)]
        input_channels = config.num_channels

        for m, d in zip(config.mult, config.down):
            output_channels = m * config.num_channels
            encoder_layers.append(ResidualBlock(input_channels, output_channels))
            input_channels = output_channels
            if d:
                encoder_layers.append(Downsample(output_channels))
        encoder_layers.extend([
            nn.GroupNorm(num_groups=32, num_channels=input_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(input_channels, config.latent_dim, kernel_size=3, stride=1, padding=1)
        ])
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        b, t, _, _, _ = x.size()
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.encoder(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)

        return x


class FrameDecoder(nn.Module):
    def __init__(self, config: FrameCnnConfig) -> None:
        super().__init__()

        assert len(config.mult) == len(config.down)
        decoder_layers = []
        output_channels = config.num_channels

        for m, d in zip(config.mult, config.down):
            input_channels = m * config.num_channels
            decoder_layers.append(ResidualBlock(input_channels, output_channels))
            output_channels = input_channels
            if d:
                decoder_layers.append(Upsample(input_channels))
        decoder_layers.reverse()
        decoder_layers.insert(0, nn.Conv2d(config.latent_dim, input_channels, kernel_size=3, stride=1, padding=1))
        decoder_layers.extend([
            nn.GroupNorm(num_groups=32, num_channels=config.num_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.num_channels, config.image_channels, kernel_size=3, stride=1, padding=1)
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        b, t, _, _, _ = x.size()
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.decoder(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups_norm: int = 32) -> None:
        super().__init__()

        self.f = nn.Sequential(
            nn.GroupNorm(num_groups_norm, in_channels), 
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups_norm, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.skip_projection = nn.Identity() if in_channels == out_channels else torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_projection(x) + self.f(x) 


class Downsample(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
