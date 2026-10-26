from cs336_basics.multihead_self_attention import multihead_self_attention
from cs336_basics.rmsnorm import rmsnorm
from cs336_basics.swiglu import SiLUFFN, SwiGLU
import torch
from torch import nn


class transformer_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device=None,
        dtype=None,
        normalization="pre",
        use_rope=True,
        ffn_type="swiglu",
    ):
        super().__init__()
        if normalization not in ("pre", "post", "none"):
            raise ValueError("normalization must be pre, post or none")
        if ffn_type not in ("swiglu", "silu"):
            raise ValueError("ffn_type must be swiglu or silu")
        self.normalization = normalization

        self.attn = multihead_self_attention(
            d_model=d_model,
            theta=theta if use_rope else None,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        if normalization == "none":
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        else:
            self.ln1 = rmsnorm(d_model, device=device, dtype=dtype)
            self.ln2 = rmsnorm(d_model, device=device, dtype=dtype)

        ffn_class = SwiGLU if ffn_type == "swiglu" else SiLUFFN
        self.ffn = ffn_class(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        if self.normalization == "post":
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x), token_positions=token_positions)

        x = x + self.ffn(self.ln2(x))

        return x

    def forward_cached(self, x, past_key_value=None):
        if self.normalization == "post":
            attention, cache = self.attn.forward_cached(x, past_key_value)
            x = self.ln1(x + attention)
            return self.ln2(x + self.ffn(x)), cache
        attention, cache = self.attn.forward_cached(self.ln1(x), past_key_value)
        x = x + attention
        return x + self.ffn(self.ln2(x)), cache
