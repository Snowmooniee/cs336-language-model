import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make_plan(
    spec,
    phase,
    data_root,
    output_root,
    learning_rate=None,
    total_tokens=None,
    batch_sizes=None,
    micro_batch_size=None,
    device="cuda",
    tf32=False,
):
    total_tokens = spec["total_tokens"] if total_tokens is None else total_tokens
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if phase != "lr" and (learning_rate is None or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Select a learning rate from actual results before running this phase")
    candidates = []
    if phase == "lr":
        candidates = [(f"lr-{lr:g}", {"max_lr": lr}) for lr in spec["learning_rates"]]
    elif phase == "batch":
        candidates = [(f"batch-{batch}", {"batch_size": batch}) for batch in (batch_sizes or spec["batch_sizes"])]
    elif phase == "ablation":
        candidates = [
            (item["name"], {key: value for key, value in item.items() if key != "name"}) for item in spec["ablations"]
        ]
    elif phase in ("baseline", "owt"):
        candidates = [(phase, {})]
    else:
        raise ValueError("Unknown experiment phase")

    jobs = []
    for name, overrides in candidates:
        config = dict(spec["baseline"])
        config["max_lr"] = learning_rate if learning_rate is not None else spec["learning_rates"][0]
        config.update(overrides)
        config["max_lr"] *= config.pop("lr_factor", 1.0)
        config["min_lr"] = config["max_lr"] * 0.1
        effective_batch = config["batch_size"]
        if effective_batch <= 0:
            raise ValueError("batch sizes must be positive")
        if micro_batch_size is not None:
            if micro_batch_size <= 0 or effective_batch % micro_batch_size:
                raise ValueError("micro_batch_size must divide every selected effective batch size")
            config["batch_size"] = micro_batch_size
            config["gradient_accumulation_steps"] = effective_batch // micro_batch_size
            name += f"-micro-{micro_batch_size}"
        tokens_per_step = effective_batch * config["context_length"]
        if phase == "batch":
            baseline_batch = spec["baseline"]["batch_size"]
            config["eval_batch_size"] = baseline_batch
            for field in ("log_interval", "eval_interval", "checkpoint_interval"):
                config[field] = max(1, round(config[field] * baseline_batch / effective_batch))
        if total_tokens % tokens_per_step:
            raise ValueError("total_tokens must be divisible by effective_batch * context_length")
        config["steps"] = total_tokens // tokens_per_step
        config["cosine_cycle_iters"] = config["steps"]
        config["warmup_iters"] = min(config["steps"] - 1, max(1, config["steps"] // 100))
        config["device"] = device
        config["tf32"] = tf32
        dataset = "owt" if phase == "owt" else "tinystories"
        if dataset == "owt":
            config["vocab_size"] = 32000
        config["train_data"] = str(Path(data_root) / dataset / "train.bin")
        config["valid_data"] = str(Path(data_root) / dataset / "valid.bin")
        name = dataset + "-" + name
        config["output_dir"] = str(Path(output_root) / name)
        jobs.append(
            {
                "name": name,
                "dataset": dataset,
                "effective_batch_size": effective_batch,
                "total_tokens": total_tokens,
                "config": config,
            }
        )
    return jobs


def training_command(config):
    command = [sys.executable, "-B", "-m", "cs336_basics.train"]
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command


def source_hash():
    digest = hashlib.sha256()
    paths = sorted((ROOT / "cs336_basics").glob("*.py")) + [Path(__file__)]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=ROOT / "experiments" / "a1.json")
    parser.add_argument("--phase", choices=["lr", "baseline", "batch", "ablation", "owt"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/tokens"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--total-tokens", type=int)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if (args.max_runs is not None and args.max_runs <= 0) or (
        args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0)
    ):
        raise ValueError("Run and time limits must be positive")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    jobs = make_plan(
        spec,
        args.phase,
        args.data_root.resolve(),
        args.output_root.resolve(),
        args.learning_rate,
        args.total_tokens,
        args.batch_sizes,
        args.micro_batch_size,
        args.device,
        args.tf32,
    )
    plan = {
        "official_commit": spec["official_commit"],
        "handout_version": spec["handout_version"],
        "source_sha256": source_hash(),
        "phase": args.phase,
        "jobs": jobs,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise FileExistsError("Existing plan differs; choose a new output root")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return

    summary_path = args.output_root / "results.json"
    results = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    completed = {item["name"] for item in results if item["status"] == "completed"}
    for job in jobs[: args.max_runs]:
        if job["name"] in completed:
            continue
        config = job["config"]
        for name in ("train_data", "valid_data"):
            if not Path(config[name]).is_file():
                raise FileNotFoundError(config[name])
        dataset_manifest = Path(config["train_data"]).parent / "dataset.json"
        if dataset_manifest.exists():
            info = json.loads(dataset_manifest.read_text(encoding="utf-8"))
            if info["vocab_size"] != config["vocab_size"]:
                raise ValueError("Prepared dataset vocabulary differs from the experiment")
        directory = Path(config["output_dir"])
        directory.mkdir(parents=True, exist_ok=False)
        console = directory / "console.log"
        start = time.perf_counter()
        with console.open("w", encoding="utf-8") as f:
            try:
                process = subprocess.run(
                    training_command(config), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=args.max_seconds
                )
                status = "completed" if process.returncode == 0 else "failed"
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                status, returncode = "timed_out", None
        if status == "failed":
            text = console.read_text(encoding="utf-8").lower()
            if "out of memory" in text:
                status = "out_of_memory"
            elif "loss is not finite" in text or "gradient norm is not finite" in text:
                status = "diverged"
        record = {
            "name": job["name"],
            "status": status,
            "returncode": returncode,
            "seconds": time.perf_counter() - start,
            "directory": str(directory),
        }
        metrics = directory / "metrics.jsonl"
        if metrics.exists():
            points = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
            record["last_metric"] = points[-1] if points else None
        results.append(record)
        summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
