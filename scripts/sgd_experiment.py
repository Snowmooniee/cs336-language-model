import argparse
import json
import time
from pathlib import Path

import torch

from cs336_basics.optimizer import SGD


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    initial = 5 * torch.randn(10, 10)
    jobs, results = [], []
    for learning_rate in args.learning_rates:
        name = f"sgd-{learning_rate:g}"
        directory = args.output_dir / name
        directory.mkdir()
        weights = torch.nn.Parameter(initial.clone())
        optimizer = SGD([weights], lr=learning_rate)
        started = time.perf_counter()
        losses = []
        with (directory / "metrics.jsonl").open("x", encoding="utf-8") as f:
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                loss = weights.square().mean()
                losses.append(loss.item())
                record = {"step": step, "train_loss": loss.item(), "elapsed_seconds": time.perf_counter() - started}
                f.write(json.dumps(record, allow_nan=False) + "\n")
                loss.backward()
                optimizer.step()
        jobs.append({"name": name, "config": {"output_dir": str(directory), "learning_rate": learning_rate}})
        results.append({"name": name, "status": "completed", "initial_loss": losses[0], "last_logged_loss": losses[-1]})
    (args.output_dir / "plan.json").write_text(
        json.dumps({"phase": "learning_rate_tuning", "seed": args.seed, "jobs": jobs}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
