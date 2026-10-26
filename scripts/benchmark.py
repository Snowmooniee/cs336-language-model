import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parameter_count(config):
    width, layers, hidden, vocab = (config[key] for key in ("d_model", "num_layers", "d_ff", "vocab_size"))
    projections = 4 * width * width
    feedforward = (3 if config.get("ffn_type", "swiglu") == "swiglu" else 2) * width * hidden
    norm = 0 if config.get("normalization", "pre") == "none" else width
    return 2 * vocab * width + layers * (projections + feedforward + 2 * norm) + norm


def measure(config, batch_size, device, warmup_steps, measure_steps, tf32):
    import torch
    from cs336_basics.optimizer import AdamW
    from cs336_basics.training_utils import cross_entropy, gradient_clipping
    from cs336_basics.transformer_lm import TransformerLM

    device = torch.device(device)
    if tf32:
        if device.type != "cuda":
            raise ValueError("tf32 requires CUDA")
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(336)
    keys = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
    model = TransformerLM(
        **{key: config[key] for key in keys},
        device=device,
        normalization=config.get("normalization", "pre"),
        use_rope=not config.get("no_rope", False),
        ffn_type=config.get("ffn_type", "swiglu"),
    )
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    started = None
    for step in range(warmup_steps + measure_steps):
        if step == warmup_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch = torch.randint(config["vocab_size"], (batch_size, config["context_length"] + 1), device=device)
        loss = cross_entropy(model(batch[:, :-1]), batch[:, 1:])
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    tokens_per_second = batch_size * config["context_length"] * measure_steps / seconds
    record = {
        "batch_size": batch_size,
        "status": "completed",
        "seconds": seconds,
        "steps": measure_steps,
        "tokens_per_second": tokens_per_second,
        "training_only_hours_for_327680000_tokens": 327680000 / tokens_per_second / 3600,
    }
    if device.type == "cuda":
        record["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
        record["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=ROOT / "experiments" / "a1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.warmup_steps < 0 or args.measure_steps <= 0 or any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("Require nonnegative warmup and positive measurement steps/batch sizes")
    config = json.loads(args.spec.read_text(encoding="utf-8"))["baseline"]
    count = parameter_count(config)
    report = {
        "config": config,
        "parameters": count,
        "fp32_weights_mib": count * 4 / 1024**2,
        "fp32_weights_gradients_adam_states_mib": count * 16 / 1024**2,
        "activation_memory_included": False,
        "executed": args.execute,
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.execute:
        import gc
        import torch

        for batch_size in args.batch_sizes:
            try:
                record = measure(config, batch_size, args.device, args.warmup_steps, args.measure_steps, args.tf32)
            except torch.OutOfMemoryError:
                record = {"batch_size": batch_size, "status": "out_of_memory"}
            report["measurements"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record), flush=True)
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if record["status"] == "out_of_memory":
                break
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
