import argparse
import cProfile
import json
import pickle
import pstats
import resource
import sys
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    special_tokens = args.special_token if args.special_token is not None else ["<|endoftext|>"]

    profiler = cProfile.Profile()
    start = time.perf_counter()
    if args.profile:
        profiler.enable()
    try:
        vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens, progress=args.progress)
    finally:
        if args.profile:
            profiler.disable()
    training_seconds = time.perf_counter() - start
    longest_token = max(vocab.values(), key=len)
    report = {
        "input": str(args.input.resolve()),
        "requested_vocab_size": args.vocab_size,
        "vocab_size": len(vocab),
        "merges": len(merges),
        "training_seconds": training_seconds,
        "peak_process_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "longest_token": repr(longest_token),
        "longest_token_bytes": len(longest_token),
        "profile_enabled": args.profile,
        "progress_enabled": args.progress,
    }
    if args.progress:
        print("Saving tokenizer artifacts...", file=sys.stderr, flush=True)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "vocab.pkl").open("xb") as f:
        pickle.dump(vocab, f)
    with (args.output_dir / "merges.pkl").open("xb") as f:
        pickle.dump(merges, f)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.profile:
        profiler.dump_stats(str(args.output_dir / "training.prof"))
        pstats.Stats(profiler).sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
