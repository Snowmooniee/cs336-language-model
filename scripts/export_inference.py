import argparse
import json
import shutil
from pathlib import Path

import torch

from cs336_basics.generation import load_model
from cs336_basics.tokenizer import Tokenizer
from scripts.prepare_data import file_hash


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_hash = file_hash(args.checkpoint)
    if args.expected_checkpoint_sha256 is not None and source_hash != args.expected_checkpoint_sha256.lower():
        raise ValueError("Source checkpoint SHA-256 mismatch")
    config_path = args.config or args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    keys = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
    inference_config = {key: config[key] for key in keys}
    inference_config.update(
        {
            "normalization": config.get("normalization", "pre"),
            "no_rope": config.get("no_rope", False),
            "ffn_type": config.get("ffn_type", "swiglu"),
        }
    )
    model = load_model(args.checkpoint, config_path, "cpu")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    if set(tokenizer.vocab) != set(range(config["vocab_size"])):
        raise ValueError("Tokenizer IDs do not match the model vocabulary")
    state = model.state_dict()
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Model contains nonfinite parameters")
    args.output_dir.mkdir(parents=True)
    weights = args.output_dir / "weights.pt"
    torch.save({"model": state}, weights)
    (args.output_dir / "config.json").write_text(json.dumps(inference_config, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(args.vocab, args.output_dir / "vocab.pkl")
    shutil.copyfile(args.merges, args.output_dir / "merges.pkl")
    exported = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if set(exported) != {"model"} or set(exported["model"]) != set(state):
        raise AssertionError("Exported state keys differ from the source model")
    for name, tensor in state.items():
        if not torch.equal(tensor, exported["model"][name]):
            raise AssertionError(f"Exported parameter differs: {name}")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        for path in sorted(args.output_dir.iterdir())
    }
    manifest = {
        "format": "torch.save dictionary containing model state only; no optimizer or RNG state",
        "source_revision": args.source_revision,
        "source_checkpoint_sha256": source_hash,
        "source_config_sha256": file_hash(config_path),
        "source_iteration": source.get("iteration"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "all_exported_parameters_equal": True,
        "torch": str(torch.__version__),
        "files": files,
        "usage": "Requires this project source and its pinned environment; see docs/delivery.md",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums = "".join(f"{file_hash(path)}  {path.name}\n" for path in sorted(args.output_dir.iterdir()))
    (args.output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
