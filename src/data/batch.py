from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch

from .segment import SegmentId


@dataclass
class Batch:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    ends: torch.LongTensor
    mask_padding: torch.BoolTensor
    segment_ids: List[SegmentId]

    def pin_memory(self) -> Batch:
        return Batch(**{k: v if k == 'segment_ids' else v.pin_memory() for k, v in self.__dict__.items()})

    def to(self, device: torch.device) -> Batch:
        return Batch(**{k: v if k == 'segment_ids' else v.to(device) for k, v in self.__dict__.items()})

