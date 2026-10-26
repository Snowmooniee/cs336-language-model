import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.optimizer import AdamW
from cs336_basics.training_utils import (
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.transformer_lm import TransformerLM


def load_tokens(path, dtype, vocab_size, context_length):
    path = Path(path)
    if path.suffix == ".npy":
        tokens = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        tokens = np.memmap(path, mode="r", dtype=dtype)

    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("Token data must be a one-dimensional integer array")
    if len(tokens) <= context_length:
        raise ValueError("Token data must contain more tokens than context_length")
    for start in range(0, len(tokens), 1_000_000):
        chunk = tokens[start : start + 1_000_000]
        if chunk.min() < 0 or chunk.max() >= vocab_size:
            raise ValueError(f"Token IDs outside [0, vocab_size): {path}")
    return tokens


def evaluate(model, dataset, batch_size, context_length, device, num_batches):
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    rng = np.random.default_rng(0)
    try:
        with torch.no_grad():
            for _ in range(num_batches):
                x, y = get_batch(dataset, batch_size, context_length, device, rng=rng)
                loss = cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Validation loss is not finite")
                total_loss += loss.item()
    finally:
        model.train(was_training)
    return total_loss / num_batches


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dtype", choices=["uint16", "uint32", "int32", "int64"], default="uint16")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--normalization", choices=["pre", "post", "none"], default="pre")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=1000)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    eval_batch_size = args.batch_size if args.eval_batch_size is None else args.eval_batch_size
    positive = [
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.steps,
        args.log_interval,
        args.eval_interval,
        args.eval_batches,
        eval_batch_size,
        args.checkpoint_interval,
    ]
    if any(value <= 0 for value in positive):
        raise ValueError("Model sizes, batch size, steps and intervals must be positive")
    if args.d_model % (2 * args.num_heads) != 0:
        raise ValueError("d_model must be divisible by 2 * num_heads for RoPE")
    if not math.isfinite(args.rope_theta) or args.rope_theta <= 0:
        raise ValueError("rope_theta must be finite and positive")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be finite and nonnegative")
    get_lr_cosine_schedule(0, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
    if args.tf32:
        if not args.device.startswith("cuda"):
            raise ValueError("--tf32 requires a CUDA device")
        torch.set_float32_matmul_precision("high")

    args.train_data = args.train_data.resolve(strict=True)
    args.valid_data = args.valid_data.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if args.train_data.samefile(args.valid_data):
        raise ValueError("Training and validation must use separate files")

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        if args.resume is None:
            raise FileExistsError("Output directory already has a run; use --resume or choose another directory")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key, value in {
            "normalization": "pre",
            "no_rope": False,
            "ffn_type": "swiglu",
            "gradient_accumulation_steps": 1,
            "tf32": False,
            "eval_batch_size": None,
        }.items():
            previous.setdefault(key, value)
        mutable = {
            "steps",
            "resume",
            "device",
            "output_dir",
            "log_interval",
            "eval_interval",
            "eval_batches",
            "checkpoint_interval",
        }
        for key in config.keys() - mutable:
            if previous[key] != config[key]:
                raise ValueError(f"Resume configuration differs: {key}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = load_tokens(args.train_data, args.data_dtype, args.vocab_size, args.context_length)
    valid_data = load_tokens(args.valid_data, args.data_dtype, args.vocab_size, args.context_length)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        normalization=args.normalization,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    )
    optimizer = AdamW(
        model.parameters(), lr=args.max_lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay
    )
    iteration = 0
    elapsed_offset = 0.0
    if args.resume is not None:
        iteration, metadata = load_checkpoint(args.resume, model, optimizer, return_metadata=True)
        elapsed_offset = metadata.get("elapsed_seconds", 0.0)
    if iteration >= args.steps:
        raise ValueError("--steps must exceed the completed iterations in the checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metrics_path = args.output_dir / "metrics.jsonl"
    if args.resume is not None and metrics_path.exists():
        lines = metrics_path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if json.loads(line)["step"] <= iteration]
        if len(kept) != len(lines):
            backup = metrics_path.with_name(f"metrics-before-resume-{time.time_ns()}.jsonl")
            metrics_path.rename(backup)
            metrics_path.write_text("".join(kept), encoding="utf-8")
    checkpoint_path = args.output_dir / "checkpoint.pt"
    start_time = time.perf_counter()
    start_iteration = iteration
    model.train()

    for step in range(iteration, args.steps):
        learning_rate = get_lr_cosine_schedule(
            step, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Training loss is not finite at step {step}")
            (loss / args.gradient_accumulation_steps).backward()
            train_loss += loss.item() / args.gradient_accumulation_steps
            del logits, loss
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        iteration = step + 1

        should_evaluate = iteration % args.eval_interval == 0 or iteration == args.steps
        if iteration % args.log_interval == 0 or should_evaluate:
            record = {
                "step": iteration,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "tokens_processed": iteration
                * args.batch_size
                * args.gradient_accumulation_steps
                * args.context_length,
            }
            if should_evaluate:
                record["validation_loss"] = evaluate(
                    model, valid_data, eval_batch_size, args.context_length, args.device, args.eval_batches
                )
                try:
                    record["validation_perplexity"] = math.exp(record["validation_loss"])
                except OverflowError:
                    record["validation_perplexity"] = None
            elapsed = time.perf_counter() - start_time
            record["elapsed_seconds"] = elapsed_offset + elapsed
            record["tokens_per_second"] = (
                (iteration - start_iteration)
                * args.batch_size
                * args.gradient_accumulation_steps
                * args.context_length
                / elapsed
            )
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        if iteration % args.checkpoint_interval == 0 or iteration == args.steps:
            save_checkpoint(
                model,
                optimizer,
                iteration,
                checkpoint_path,
                metadata={"elapsed_seconds": elapsed_offset + time.perf_counter() - start_time},
            )


if __name__ == "__main__":
    main()
