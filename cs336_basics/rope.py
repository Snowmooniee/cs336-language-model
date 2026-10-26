import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k

        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1 / (theta**powers)

        t = torch.arange(max_seq_len, device=device).float()

        freqs_matrix = torch.outer(t, freqs)

        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output
