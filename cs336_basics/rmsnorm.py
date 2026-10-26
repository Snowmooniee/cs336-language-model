import torch
from torch import nn


class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        ms = x.pow(2).mean(-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)

        result = (x / rms) * self.weight

        return result.to(in_dtype)
