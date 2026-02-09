from __future__ import annotations

import torch
import torch.nn as nn


class MyDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.001:
            mask = torch.rand(x.size(), device=x.device).lt(self.p)
            x = x.masked_fill(mask, 0) / (1 - self.p)
        return x


def print_info(*inp, sep: str = " ", islog: bool = True):
    # Keep signature compatible with historical calls.
    _ = islog
    print(*inp, sep=sep)


def size2MB(size_, type_size: int = 4) -> float:
    num = 1
    for s in size_:
        num *= s
    return num * type_size / 1000 / 1000
