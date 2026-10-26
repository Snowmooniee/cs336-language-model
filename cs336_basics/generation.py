import json
import math
import time
from pathlib import Path

import torch

from cs336_basics.transformer_lm import TransformerLM


def sample_next_token(logits, temperature=1.0, top_p=1.0, generator=None):
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a nonempty vocabulary vector")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if temperature == 0:
        return int(logits.argmax().item())

    scores = logits if logits.dtype == torch.float64 else logits.float()
    centered = scores - scores.max()
    probabilities = torch.softmax(centered / temperature, dim=-1)
    sorted_probabilities, indices = probabilities.sort(descending=True)
    mass_before = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities
    sorted_probabilities = sorted_probabilities.masked_fill(mass_before >= top_p, 0)
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum()
    selected = torch.multinomial(sorted_probabilities, 1, generator=generator)
    return int(indices[selected].item())


@torch.no_grad()
def generate(
    model,
    prompt_ids,
    max_new_tokens=256,
    temperature=1.0,
    top_p=1.0,
    eos_token_id=None,
    generator=None,
    use_kv_cache=False,
    stats=None,
):
    if not prompt_ids or max_new_tokens < 0:
        raise ValueError("Require a nonempty prompt and max_new_tokens >= 0")
    device = next(model.parameters()).device
    if use_kv_cache and not callable(getattr(model, "forward_cached", None)):
        raise ValueError("This model does not implement cached inference")
    ids = list(prompt_ids)
    was_training = model.training
    model.eval()
    cache, cached_length, rebuilds = None, 0, 0
    first_token_seconds = None
    if stats is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter() if stats is not None else None
    try:
        for _ in range(max_new_tokens):
            if use_kv_cache:
                if cache is None or cached_length >= model.context_length:
                    if cache is not None:
                        rebuilds += 1
                    context = torch.tensor([ids[-model.context_length :]], dtype=torch.long, device=device)
                    logits, cache = model.forward_cached(context)
                    cached_length = context.shape[-1]
                else:
                    context = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                    logits, cache = model.forward_cached(context, cache)
                    cached_length += 1
                logits = logits[0, -1]
            else:
                context = torch.tensor([ids[-model.context_length :]], dtype=torch.long, device=device)
                logits = model(context)[0, -1]
            token = sample_next_token(logits, temperature, top_p, generator)
            ids.append(token)
            if stats is not None and first_token_seconds is None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                first_token_seconds = time.perf_counter() - started
            if token == eos_token_id:
                break
    finally:
        model.train(was_training)
    if stats is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        count = len(ids) - len(prompt_ids)
        decode_seconds = elapsed - first_token_seconds if count else 0.0
        stats.update(
            {
                "generated_tokens": count,
                "elapsed_seconds": elapsed,
                "time_to_first_token_seconds": first_token_seconds,
                "decode_seconds": decode_seconds,
                "decode_tokens_per_second": (count - 1) / decode_seconds if count > 1 and decode_seconds > 0 else None,
                "cache_bytes": sum(tensor.numel() * tensor.element_size() for pair in cache for tensor in pair)
                if cache
                else 0,
                "cache_tokens": cached_length,
                "context_rebuilds": rebuilds,
            }
        )
    return ids


def load_model(checkpoint_path, config_path=None, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path) if config_path is not None else checkpoint_path.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    keys = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
    model = TransformerLM(
        **{key: config[key] for key in keys},
        device=device,
        normalization=config.get("normalization", "pre"),
        use_rope=not config.get("no_rope", False),
        ffn_type=config.get("ffn_type", "swiglu"),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
