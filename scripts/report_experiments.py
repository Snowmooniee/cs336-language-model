import argparse
import csv
import json
from pathlib import Path


def collect_points(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    summary_path = root / "results.json"
    results = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    statuses = {item["name"]: item["status"] for item in results}
    points, summary = [], []
    for job in plan["jobs"]:
        config = job["config"]
        metrics = Path(config["output_dir"]) / "metrics.jsonl"
        records = (
            [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()] if metrics.exists() else []
        )
        for record in records:
            points.append({"run": job["name"], **record})
        validation = [row for row in records if "validation_loss" in row]
        summary.append(
            {
                "run": job["name"],
                "status": statuses.get(job["name"], "not_run"),
                "last_step": records[-1]["step"] if records else None,
                "last_train_loss": records[-1]["train_loss"] if records else None,
                "last_validation_loss": validation[-1]["validation_loss"] if validation else None,
                "max_learning_rate": config.get("max_lr", config.get("learning_rate")),
                "normalization": config.get("normalization"),
                "ffn_type": config.get("ffn_type"),
                "physical_batch_size": config.get("batch_size"),
                "effective_batch_size": job.get("effective_batch_size"),
                "no_rope": config.get("no_rope"),
                "d_ff": config.get("d_ff"),
            }
        )
    return points, summary


def collect_bundle(path, phase):
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = bundle["runs"]
    if len({run["name"] for run in runs}) != len(runs):
        raise ValueError("Published run names must be unique")
    chosen = [run for run in runs if run["phase"] == phase]
    if phase == "ablation":
        baseline = bundle["ablation_baseline"]
        chosen = [next(run for run in runs if run["name"] == baseline), *chosen]
    if not chosen:
        raise ValueError("No published runs found for this phase")
    points, summary = [], []
    for run in chosen:
        records, config = run["metrics"], run["config"]
        points.extend({"run": run["name"], **record} for record in records)
        validation = [record for record in records if "validation_loss" in record]
        summary.append(
            {
                "run": run["name"],
                "status": run["status"],
                "last_step": records[-1]["step"] if records else None,
                "last_train_loss": records[-1]["train_loss"] if records else None,
                "last_validation_loss": validation[-1]["validation_loss"] if validation else None,
                "max_learning_rate": config["max_lr"],
                "normalization": config["normalization"],
                "ffn_type": config["ffn_type"],
                "physical_batch_size": config["batch_size"],
                "effective_batch_size": run["effective_batch_size"],
                "no_rope": config["no_rope"],
                "d_ff": config["d_ff"],
            }
        )
    return points, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run-root", type=Path)
    inputs.add_argument("--bundle", type=Path)
    parser.add_argument("--phase", choices=["lr", "batch", "ablation"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.bundle is not None:
        if args.phase is None:
            parser.error("--bundle requires --phase")
        points, summary = collect_bundle(args.bundle, args.phase)
    else:
        points, summary = collect_points(args.run_root)
    if args.plot:
        if not points:
            raise ValueError("No recorded metrics are available to plot")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "summary.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["run", "status"])
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / "curves.csv").open("x", newline="", encoding="utf-8") as f:
        keys = sorted({key for point in points for key in point})
        writer = csv.DictWriter(f, fieldnames=keys or ["run", "step"])
        writer.writeheader()
        writer.writerows(points)
    if args.plot:
        x_keys = ["step", "elapsed_seconds"]
        labels = ["Gradient steps", "Active wall-clock seconds"]
        if any("tokens_processed" in point for point in points):
            x_keys.append("tokens_processed")
            labels.append("Training tokens")
        figure, axes = plt.subplots(1, len(x_keys), figsize=(5 * len(x_keys), 4))
        for run in dict.fromkeys(point["run"] for point in points):
            records = [point for point in points if point["run"] == run]
            validation = [point for point in records if "validation_loss" in point]
            records = validation or records
            key = "validation_loss" if validation else "train_loss"
            for axis, x_key in zip(axes, x_keys):
                chosen = [point for point in records if x_key in point]
                axis.plot([point[x_key] for point in chosen], [point[key] for point in chosen], label=run)
        for axis, label in zip(axes, labels):
            axis.set_xlabel(label)
            axis.set_ylabel("Validation loss")
            axis.grid(alpha=0.25)
        if points:
            axes[0].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(args.output_dir / "learning_curves.png", dpi=180)
        plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
