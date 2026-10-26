import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.rope import RoPE
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention


class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.attention = ScaledDotProductAttention()

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def _project(self, x, token_positions=None):
        s = x.shape[-2]

        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(s, device=x.device)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        return q, k, v

    def _attend(self, q, k, v, mask):
        attn_out = self.attention(q, k, v, mask=mask)
        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")
        return self.output_proj(attn_out)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        q, k, v = self._project(x, token_positions)
        s = x.shape[-2]
        mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))
        return self._attend(q, k, v, mask)

    def forward_cached(self, x, past_key_value=None):
        past_length = 0
        if past_key_value is not None:
            if len(past_key_value) != 2 or not all(isinstance(tensor, torch.Tensor) for tensor in past_key_value):
                raise ValueError("Each cache must contain a key and value tensor")
            old_k, old_v = past_key_value
            if old_k.ndim != x.ndim + 1 or old_k.shape != old_v.shape:
                raise ValueError("Cached key/value shape mismatch")
            past_length = old_k.shape[-2]
            expected = (*x.shape[:-2], self.num_heads, past_length, self.d_k)
            if old_k.shape != expected:
                raise ValueError("Cache shape does not match the input batch and heads")

        positions = torch.arange(past_length, past_length + x.shape[-2], device=x.device)
        q, k, v = self._project(x, positions)
        if past_key_value is not None:
            if old_k.dtype != k.dtype or old_v.dtype != v.dtype:
                raise ValueError("Cache dtype does not match the projected keys and values")
            if old_k.device != k.device or old_v.device != v.device:
                raise ValueError("Cache device does not match the input")
            k = torch.cat((old_k, k), dim=-2)
            v = torch.cat((old_v, v), dim=-2)
        mask = torch.arange(k.shape[-2], device=x.device)[None, :] <= positions[:, None]
        return self._attend(q, k, v, mask), (k, v)
