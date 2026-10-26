import argparse
import hashlib
import json
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer
from scripts.tokenizer_experiments import encode_file


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--valid-text", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.train_text.samefile(args.valid_text):
        raise ValueError("Training and validation must use separate files")
    train_hash = file_hash(args.train_text)
    valid_hash = file_hash(args.valid_text)
    if train_hash == valid_hash:
        raise ValueError("Training and validation content is identical")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    report = {
        "dtype": "<u2",
        "vocab_size": len(tokenizer.vocab),
        "vocab_sha256": file_hash(args.vocab),
        "merges_sha256": file_hash(args.merges),
        "splits": {},
    }
    args.output_dir.mkdir(parents=True)
    for name, source, digest in (("train", args.train_text, train_hash), ("valid", args.valid_text, valid_hash)):
        target = args.output_dir / (name + ".bin")
        count = encode_file(tokenizer, source, target)
        report["splits"][name] = {
            "source": str(source.resolve()),
            "source_sha256": digest,
            "tokens": count,
            "bytes": target.stat().st_size,
            "sha256": file_hash(target),
        }
    (args.output_dir / "dataset.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
