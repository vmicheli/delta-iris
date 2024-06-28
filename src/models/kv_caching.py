from typing import Tuple

import numpy as np
import torch


class Cache:
    def __init__(self, num_samples: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        self._n, self._cache, self._size = num_samples, None, None
        self._reset = lambda n: torch.empty(n, max_tokens, embed_dim, device=device)  # (B, T, E)
        self.reset()

    @property
    def shape(self) -> Tuple[int, int, int]:
        n, _, embed_dim = self._cache.shape

        return n, self._size, embed_dim

    def reset(self) -> None:
        self._cache = self._reset(self._n)
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        assert mask.ndim == 1 and mask.shape[0] == self.shape[0]
        self._cache = self._cache[mask]
        self._n = self._cache.shape[0]

    def get(self) -> torch.Tensor:
        return self._cache[:, :self._size, :]

    def update(self, x: torch.Tensor) -> None:
        assert (x.ndim == self._cache.ndim) and all([x.size(i) == self._cache.size(i) for i in (0, 2)])
        assert self._size + x.size(1) <= self._cache.shape[1]
        self._cache = AssignWithoutInplaceCheck.apply(self._cache, x, 1, self._size, self._size + x.size(1))
        self._size += x.size(1)


class KVCache:
    def __init__(self, n: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        self._k_cache = Cache(n, max_tokens, embed_dim, device)
        self._v_cache = Cache(n, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._k_cache.shape

    def reset(self) -> None:
        self._k_cache.reset()
        self._v_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor):
        self._k_cache.update(k)
        self._v_cache.update(v)


class KeysValues:
    def __init__(self, n: int, max_tokens: int, embed_dim: int, num_layers: int, device: torch.device) -> None:
        self._keys_values = tuple([KVCache(n, max_tokens, embed_dim, device) for _ in range(num_layers)])

    def __getitem__(self, key: int) -> KVCache:
        return self._keys_values[key]

    def __len__(self):
        return len(self._keys_values)

    @property
    def size(self):
        return self._keys_values[0].shape[1]

    def reset(self) -> None:
        for kv_cache in self._keys_values:
            kv_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        for kv_cache in self._keys_values:
            kv_cache.prune(mask)


class AssignWithoutInplaceCheck(torch.autograd.Function):
    """
    Inspired from : https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/4
    Warning : do not use it to overwrite a slice twice.
    """

    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice]:
        return tuple([slice(None), ] * dim + [slice(start, stop)])

    @staticmethod
    def forward(ctx, input: torch.Tensor, value: torch.Tensor, dim: int, start: int, stop: int) -> torch.Tensor:
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        input.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = value
        return input

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor]:
        return grad_out, grad_out[AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)], None, None, None
