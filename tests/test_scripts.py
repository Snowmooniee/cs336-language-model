import json

import numpy as np
import pytest

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.bpe import train_bpe
from scripts.tokenizer_experiments import encode_file
from scripts.train_bpe import main


def test_bpe_command_writes_reloadable_artifacts(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("abab abab 猪🙂<|endoftext|>abac", encoding="utf-8")
    output = tmp_path / "bpe"
    args = ["--input", str(source), "--output-dir", str(output), "--vocab-size", "270", "--profile"]
    main(args)

    tokenizer = Tokenizer.from_files(output / "vocab.pkl", output / "merges.pkl", ["<|endoftext|>"])
    text = source.read_text(encoding="utf-8")
    assert tokenizer.decode(tokenizer.encode(text)) == text
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["vocab_size"] == len(tokenizer.vocab) <= 270
    assert report["training_seconds"] >= 0
    assert report["profile_enabled"] is True
    assert (output / "training.prof").is_file()
    with pytest.raises(FileExistsError):
        main(args)


def test_bpe_progress_reports_stages_and_preserves_results(tmp_path, capsys):
    source = tmp_path / "corpus.txt"
    source.write_text("aaaa abab 猪🙂<|endoftext|>abac", encoding="utf-8")
    expected_vocab, expected_merges = train_bpe(source, 270, ["<|endoftext|>"])
    output = tmp_path / "bpe"
    main(["--input", str(source), "--output-dir", str(output), "--vocab-size", "270", "--progress"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["progress_enabled"] is True
    for stage in ("Reading training text", "Pre-tokenization", "BPE merges", "Saving tokenizer artifacts"):
        assert stage in captured.err
    tokenizer = Tokenizer.from_files(output / "vocab.pkl", output / "merges.pkl", ["<|endoftext|>"])
    assert tokenizer.vocab == expected_vocab
    assert tokenizer.merges == expected_merges


def test_encoding_writes_uint16_stream_without_overwriting(tmp_path):
    tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [], ["<|endoftext|>"])
    source = tmp_path / "text.txt"
    text = "hello 猪🙂\n<|endoftext|>world"
    source.write_text(text, encoding="utf-8")
    target = tmp_path / "tokens.bin"

    count = encode_file(tokenizer, source, target)
    ids = np.fromfile(target, dtype="<u2")
    assert len(ids) == count
    assert tokenizer.decode(ids.tolist()) == text
    assert not target.with_suffix(".bin.partial").exists()
    with pytest.raises(FileExistsError):
        encode_file(tokenizer, source, target)


def test_published_bundle_reuses_control_and_preserves_failed_runs(tmp_path):
    from scripts.report_experiments import collect_bundle, main as report

    config = dict(max_lr=0.001, normalization="pre", ffn_type="swiglu", batch_size=32, no_rope=False, d_ff=1344)
    metric = dict(step=40000, train_loss=1.28, validation_loss=1.38, tokens_processed=327680000)
    runs = [
        dict(
            name=name,
            phase=phase,
            status=status,
            config=config,
            effective_batch_size=32,
            metrics=[] if status == "out_of_memory" else [metric],
        )
        for name, phase, status in [
            ("control", "batch", "completed"),
            ("oom", "batch", "out_of_memory"),
            ("nope", "ablation", "completed"),
        ]
    ]
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(dict(runs=runs, ablation_baseline="control")))
    points, summary = collect_bundle(path, "ablation")
    assert [row["run"] for row in summary] == ["control", "nope"]
    assert len(points) == 2
    _, batches = collect_bundle(path, "batch")
    assert batches[1]["status"] == "out_of_memory" and batches[1]["last_step"] is None
    report(["--bundle", str(path), "--phase", "batch", "--output-dir", str(tmp_path / "report")])
    assert "out_of_memory" in (tmp_path / "report/summary.csv").read_text()
    with pytest.raises(FileExistsError):
        report(["--bundle", str(path), "--phase", "batch", "--output-dir", str(tmp_path / "report")])
    path.write_text(json.dumps(dict(runs=[runs[0], runs[0]])))
    with pytest.raises(ValueError, match="unique"):
        collect_bundle(path, "batch")
