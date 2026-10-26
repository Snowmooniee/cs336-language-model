import math
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch


def cross_entropy(inputs, targets):
    if inputs.shape[:-1] != targets.shape or targets.numel() == 0:
        raise ValueError("Targets must match the non-vocabulary dimensions and be nonempty")
    if inputs.dtype in (torch.float16, torch.bfloat16):
        inputs = inputs.float()

    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < 0 or not 0 <= warmup_iters < cosine_cycle_iters:
        raise ValueError("Require it >= 0 and 0 <= warmup_iters < cosine_cycle_iters")
    if not (
        math.isfinite(max_learning_rate)
        and math.isfinite(min_learning_rate)
        and 0 <= min_learning_rate <= max_learning_rate
    ):
        raise ValueError("Require finite learning rates with 0 <= min <= max")
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)


@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm):
    if not math.isfinite(max_l2_norm) or max_l2_norm < 0:
        raise ValueError("max_l2_norm must be finite and nonnegative")
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return

    norms = [torch.linalg.vector_norm(g.float()) for g in gradients]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    if not torch.isfinite(total_norm):
        raise RuntimeError("Gradient norm is not finite")
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for gradient in gradients:
            gradient.mul_(scale)


def get_batch(dataset, batch_size, context_length, device, rng=None):
    if dataset.ndim != 1 or not np.issubdtype(dataset.dtype, np.integer):
        raise ValueError("Dataset must be a one-dimensional integer array")
    if batch_size <= 0 or context_length <= 0 or len(dataset) <= context_length:
        raise ValueError("Require positive sizes and more tokens than context_length")

    num_starts = len(dataset) - context_length
    if rng is None:
        starts = np.random.randint(0, num_starts, size=batch_size)
    else:
        starts = rng.integers(0, num_starts, size=batch_size)

    sequences = []
    for start in starts:
        sequences.append(dataset[start : start + context_length + 1])
    batch = torch.tensor(np.stack(sequences), dtype=torch.long, device=device)
    return batch[:, :-1], batch[:, 1:]


def save_checkpoint(model, optimizer, iteration, out, metadata=None):
    numpy_state = np.random.get_state()
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "metadata": metadata if metadata is not None else {},
    }
    if any(p.is_cuda for p in model.parameters()):
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    if not isinstance(out, (str, PathLike)):
        torch.save(checkpoint, out)
        return

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(src, model, optimizer, return_metadata=False):
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "numpy_rng_state" in checkpoint:
        state = checkpoint["numpy_rng_state"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    if "cuda_rng_state" in checkpoint and any(p.is_cuda for p in model.parameters()):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if return_metadata:
        return checkpoint["iteration"], checkpoint.get("metadata", {})
    return checkpoint["iteration"]
