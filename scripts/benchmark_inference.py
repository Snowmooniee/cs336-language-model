import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

import torch

from cs336_basics.generation import generate, load_model
from cs336_basics.tokenizer import Tokenizer


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text-file", type=Path)
    inputs.add_argument("--cases-file", type=Path)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-context-rebuilds", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.prompt_lengths) <= 0 or args.new_tokens < 2 or args.repeats < 2 or args.warmup < 0 or args.threads <= 0:
        raise ValueError("Require positive prompt lengths/threads, >=2 new tokens/repeats, and nonnegative warmup")
    torch.set_num_threads(args.threads)
    torch.manual_seed(336)
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    model = load_model(args.checkpoint, args.config, args.device)
    if set(tokenizer.vocab) != set(range(model.lm_head.weight.shape[0])):
        raise ValueError("Tokenizer/model vocabulary mismatch")
    if args.cases_file is None:
        text = args.text_file.read_text(encoding="utf-8")
        cases = [
            {"name": f"prefix-{length}", "text": text, "prompt_tokens": length, "new_tokens": args.new_tokens}
            for length in args.prompt_lengths
        ]
    else:
        cases = json.loads(args.cases_file.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Require a nonempty list of benchmark cases")
    names = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("text"), str):
            raise ValueError("Each case must be an object containing text")
        if not isinstance(case.get("name"), str) or not case["name"] or case["name"] in names:
            raise ValueError("Case names must be nonempty and unique")
        names.append(case["name"])
        case.setdefault("new_tokens", args.new_tokens)
        if type(case["prompt_tokens"]) is not int or type(case["new_tokens"]) is not int:
            raise ValueError("Case token budgets must be integers")
        if not 0 < case["prompt_tokens"] <= model.context_length or case["new_tokens"] < 2:
            raise ValueError("Case prompt must fit the context and new_tokens must be >=2")
        case["prompt_ids"] = tokenizer.encode(case["text"])[: case["prompt_tokens"]]
        if len(case["prompt_ids"]) != case["prompt_tokens"]:
            raise ValueError("Text contains fewer tokens than the requested prompt")
        if case["prompt_tokens"] + case["new_tokens"] - 1 > model.context_length and not args.allow_context_rebuilds:
            raise ValueError("This benchmark excludes context rebuilds unless --allow-context-rebuilds is specified")

    device = next(model.parameters()).device
    samples, checked_cases = [], []
    max_error = 0.0
    for case in cases:
        length, new_tokens, prompt = case["prompt_tokens"], case["new_tokens"], case["prompt_ids"]
        tokens = torch.tensor([prompt], device=device)
        with torch.no_grad():
            expected = model(tokens)
            cache, pieces = None, []
            for chunk in tokens.split(16, dim=-1):
                logits, cache = model.forward_cached(chunk, cache)
                pieces.append(logits)
            actual = torch.cat(pieces, dim=1)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            case_error = (actual - expected).abs().max().item()
            max_error = max(max_error, case_error)
        del expected, actual, cache, pieces, logits
        reference = None
        for cached in (False, True):
            for _ in range(args.warmup):
                generate(model, prompt, new_tokens, temperature=0, use_kv_cache=cached)
        for repeat in range(args.repeats):
            for cached in (False, True) if repeat % 2 == 0 else (True, False):
                stats = {}
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                ids = generate(model, prompt, new_tokens, temperature=0, use_kv_cache=cached, stats=stats)
                if reference is None:
                    reference = ids
                if ids != reference:
                    raise AssertionError("Cached and uncached greedy output differ")
                rebuilds = max(0, length + new_tokens - 1 - model.context_length) if cached else 0
                if stats["context_rebuilds"] != rebuilds:
                    raise AssertionError("Context rebuild count does not match the requested workload")
                row = {
                    "case": case["name"],
                    "prompt_tokens": length,
                    "new_tokens": new_tokens,
                    "repeat": repeat,
                    "kv_cache": cached,
                    "greedy_ids_match": True,
                    **stats,
                }
                if device.type == "cuda":
                    row["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
                samples.append(row)
                print(json.dumps(row), flush=True)
        checked_cases.append(
            {
                "name": case["name"],
                "prompt_tokens": length,
                "new_tokens": new_tokens,
                "maximum_prefill_logit_absolute_error": case_error,
                "prompt_token_ids": prompt,
                "generated_token_ids": reference[length:],
            }
        )

    summaries = []
    for case in checked_cases:
        pair = {}
        for cached in (False, True):
            rows = [row for row in samples if row["case"] == case["name"] and row["kv_cache"] == cached]
            pair[cached] = {
                "median_decode_tokens_per_second": statistics.median(row["decode_tokens_per_second"] for row in rows),
                "median_first_token_ms": 1000 * statistics.median(row["time_to_first_token_seconds"] for row in rows),
                "median_total_seconds": statistics.median(row["elapsed_seconds"] for row in rows),
                "max_cache_bytes": max(row["cache_bytes"] for row in rows),
                "context_rebuilds": max(row["context_rebuilds"] for row in rows),
            }
        summaries.append(
            {
                "case": case["name"],
                "prompt_tokens": case["prompt_tokens"],
                "new_tokens": case["new_tokens"],
                "uncached": pair[False],
                "cached": pair[True],
                "decode_speedup": pair[True]["median_decode_tokens_per_second"]
                / pair[False]["median_decode_tokens_per_second"],
            }
        )
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = platform.processor()
    if cpu_info.exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu_info.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_model,
        )
    report = {
        "scope": "Actual checkpoint, batch1 greedy generation; fixed cases; EOS disabled for fixed output budgets",
        "device": str(device),
        "cpu_model": cpu_model,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "dtype": str(next(model.parameters()).dtype),
        "threads": args.threads,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "new_tokens": cases[0]["new_tokens"] if len({case["new_tokens"] for case in cases}) == 1 else None,
        "context_length": model.context_length,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "vocab_sha256": file_hash(args.vocab),
        "merges_sha256": file_hash(args.merges),
        "prompt_text_sha256": file_hash(args.text_file) if args.text_file else None,
        "cases_file_sha256": file_hash(args.cases_file) if args.cases_file else None,
        "maximum_logit_absolute_error": max_error,
        "all_greedy_ids_match": True,
        "precision": torch.get_float32_matmul_precision(),
        "timing_definition": "First-token time includes prefill/selection; decode excludes first token; rebuilds included when enabled",
        "allow_context_rebuilds": args.allow_context_rebuilds,
        "checked_cases": checked_cases,
        "summary": summaries,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
    print(json.dumps(summaries, indent=2))
    return report


if __name__ == "__main__":
    main()
