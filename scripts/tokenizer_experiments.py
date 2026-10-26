import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from cs336_basics.tokenizer import Tokenizer

END = "<|endoftext|>"
SEED = 336


def iter_documents(path):
    buffer = ""

    with open(path, encoding="utf-8", newline="") as f:
        while True:
            text = f.read(1024 * 1024)
            if not text:
                break

            pieces = (buffer + text).split(END)
            buffer = pieces.pop()

            for document in pieces:
                if document:
                    yield document + END

    if buffer:
        yield buffer


def sample_documents(path, seed=SEED):
    rng = random.Random(seed)
    sample = []

    for index, document in enumerate(iter_documents(path)):
        if index < 10:
            sample.append((index, document))
        else:
            position = rng.randrange(index + 1)

            if position < 10:
                sample[position] = (index, document)

    if len(sample) < 10:
        raise ValueError(f"Fewer than 10 nonempty documents: {path}")

    return sample


def measure(tokenizer, sample):
    total_bytes = sum(len(text.encode("utf-8")) for _, text in sample)

    tokenizer.encode(sample[0][1])
    times = []

    for _ in range(3):
        start = time.perf_counter()

        encoded = [tokenizer.encode(text) for _, text in sample]

        times.append(time.perf_counter() - start)

    seconds = sorted(times)[1]
    total_tokens = sum(len(ids) for ids in encoded)

    for (_, text), ids in zip(sample, encoded):
        assert tokenizer.decode(ids) == text

    throughput = total_bytes / seconds

    return {
        "document_indices_0based": [index for index, _ in sample],
        "encoded_documents": encoded,
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "encoding_seconds_median": seconds,
        "bytes_per_second": throughput,
        "pile_825GB_hours": 825 * 10**9 / throughput / 3600,
    }


def encode_file(tokenizer, source, target):
    if min(tokenizer.vocab) < 0 or max(tokenizer.vocab) > 65535:
        raise ValueError("Vocabulary IDs do not fit uint16")

    if target.exists():
        raise FileExistsError(target)

    temporary = target.with_suffix(target.suffix + ".partial")
    batch = []
    token_count = 0

    with open(source, encoding="utf-8", newline="") as f:
        with open(temporary, "xb") as out:
            for token_id in tokenizer.encode_iterable(f):
                batch.append(token_id)

                if len(batch) == 65536:
                    np.asarray(batch, dtype="<u2").tofile(out)
                    token_count += len(batch)
                    batch.clear()

            if batch:
                np.asarray(batch, dtype="<u2").tofile(out)
                token_count += len(batch)

    temporary.rename(target)
    return token_count


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--encode", action="store_true")
    args = parser.parse_args(argv)
    data = args.data_dir
    ts_dir = args.results_dir / "tinystories-train"
    owt_dir = args.results_dir / "owt-train"

    ts = Tokenizer.from_files(
        ts_dir / "vocab.pkl",
        ts_dir / "merges.pkl",
        [END],
    )

    owt = Tokenizer.from_files(
        owt_dir / "vocab.pkl",
        owt_dir / "merges.pkl",
        [END],
    )

    if len(ts.vocab) != 10000 or len(owt.vocab) != 32000:
        raise ValueError("Expected TinyStories 10K and OWT 32K vocabularies")

    ts_sample = sample_documents(data / "TinyStoriesV2-GPT4-valid.txt", args.seed)
    owt_sample = sample_documents(data / "owt_valid.txt", args.seed)

    results = {
        "ts_on_ts": measure(ts, ts_sample),
        "owt_on_owt": measure(owt, owt_sample),
        "ts_on_owt": measure(ts, owt_sample),
    }

    run_name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.results_dir / "tokenizer-experiments" / run_name
    output.mkdir(parents=True, exist_ok=False)

    report = {
        "seed": args.seed,
        "official_commit": "a158843b20107949f1a8d7df1b05cd33b9166712",
        "data_directory": str(data),
        "tokenizer_directories": [
            str(ts_dir),
            str(owt_dir),
        ],
        "results": results,
        "full_encoding": {},
    }

    with open(output / "report.json", "x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for name, stats in results.items():
        print(name)
        print("bytes/token:", stats["bytes_per_token"])
        print("bytes/second:", stats["bytes_per_second"])
        print(
            "Estimated Pile hours:",
            stats["pile_825GB_hours"],
        )

    if args.encode:
        datasets = [
            (
                "ts_train",
                ts,
                data / "TinyStoriesV2-GPT4-train.txt",
            ),
            (
                "ts_valid",
                ts,
                data / "TinyStoriesV2-GPT4-valid.txt",
            ),
            ("owt_train", owt, data / "owt_train.txt"),
            ("owt_valid", owt, data / "owt_valid.txt"),
        ]

        for name, tokenizer, path in datasets:
            target = output / f"{name}.bin"
            count = encode_file(tokenizer, path, target)

            report["full_encoding"][name] = {
                "source": str(path),
                "output": str(target),
                "dtype": "<u2",
                "tokens": count,
                "output_bytes": target.stat().st_size,
            }

            with open(output / "report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(name, count, "tokens")

    print("Results:", output)


if __name__ == "__main__":
    main()
