import torch
from torch import nn
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import rmsnorm
from cs336_basics.transformer_block import transformer_block


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
        normalization="pre",
        use_rope=True,
        ffn_type="swiglu",
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
                normalization=normalization,
                use_rope=use_rope,
                ffn_type=ffn_type,
            )
            self.layers.append(block)

        self.ln_final = rmsnorm(d_model, device=device, dtype=dtype) if normalization != "none" else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("Sequence length exceeds context_length")

        x = self.token_embeddings(token_ids)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def forward_cached(self, token_ids, past_key_values=None):
        """Return logits and per-layer RoPE-transformed keys and values for inference."""
        if token_ids.ndim < 1 or token_ids.shape[-1] == 0:
            raise ValueError("Cached inference requires a nonempty token sequence")
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            past_length = 0
        else:
            if len(past_key_values) != len(self.layers):
                raise ValueError("Cache must contain one entry per layer")
            for cache in past_key_values:
                if cache is None or len(cache) != 2 or not all(isinstance(tensor, torch.Tensor) for tensor in cache):
                    raise ValueError("Each layer cache must contain key and value tensors")
                if cache[0].ndim < 2:
                    raise ValueError("Invalid cache shape")
            past_length = past_key_values[0][0].shape[-2] if past_key_values else 0
            if any(cache[0].shape[-2] != past_length for cache in past_key_values):
                raise ValueError("All layer caches must have the same sequence length")
        if past_length + token_ids.shape[-1] > self.context_length:
            raise ValueError("Cached sequence length exceeds context_length; rebuild the window")
        x = self.token_embeddings(token_ids)
        caches = []
        for block, cache in zip(self.layers, past_key_values):
            x, updated = block.forward_cached(x, cache)
            caches.append(updated)
        return self.lm_head(self.ln_final(x)), caches
