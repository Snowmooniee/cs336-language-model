import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cs336_basics.generation import generate, load_model, sample_next_token
from cs336_basics.optimizer import AdamW
from cs336_basics.rmsnorm import rmsnorm
from cs336_basics.train import main as train
from cs336_basics.training_utils import save_checkpoint
from cs336_basics.transformer_lm import TransformerLM
from scripts.benchmark import parameter_count
from scripts.generate import main as generate_command
from scripts.prepare_data import file_hash, main as prepare_data
from scripts.report_experiments import collect_points
from scripts.run_experiments import make_plan, main as run_experiments


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_nucleus_includes_the_token_crossing_threshold():
    logits = torch.tensor([0.4, 0.35, 0.25]).log()
    generator = torch.Generator().manual_seed(42)
    samples = [sample_next_token(logits, top_p=0.5, generator=generator) for _ in range(100)]
    assert set(samples) == {0, 1}
    assert sample_next_token(logits, top_p=0.3) == 0
    assert sample_next_token(logits, temperature=0) == 0
    with pytest.raises(ValueError):
        sample_next_token(logits, top_p=0)
    with pytest.raises(ValueError):
        sample_next_token(logits, temperature=-1)


def test_sampling_is_reproducible_with_explicit_generator():
    logits = torch.tensor([1.0, 2.0, 3.0])
    a = torch.Generator().manual_seed(7)
    b = torch.Generator().manual_seed(7)
    assert [sample_next_token(logits, generator=a) for _ in range(20)] == [
        sample_next_token(logits, generator=b) for _ in range(20)
    ]


class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0, 1.0, 3.0]))
        self.context_length = 3
        self.lengths = []

    def forward(self, ids):
        self.lengths.append(ids.shape[-1])
        return self.weight.expand(ids.shape[0], ids.shape[1], -1)


def test_generation_stops_at_eos_and_respects_context_window():
    model = FixedModel()
    assert generate(model, [0], 10, temperature=0, eos_token_id=2) == [0, 2]
    assert model.training
    ids = generate(model, [0, 1, 0, 1], 6, temperature=0)
    assert len(ids) == 10
    assert max(model.lengths) <= model.context_length
    assert generate(model, [0], 0) == [0]


@pytest.mark.parametrize("normalization", ["pre", "post", "none"])
@pytest.mark.parametrize("ffn_type", ["swiglu", "silu"])
def test_architecture_variants_have_registered_gradients(normalization, ffn_type):
    config = dict(
        vocab_size=16,
        context_length=8,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        rope_theta=10000.0,
        normalization=normalization,
        ffn_type=ffn_type,
    )
    model = TransformerLM(**config, use_rope=False)
    output = model(torch.tensor([[1, 2, 3]]))
    output.square().mean().backward()
    assert output.shape == (1, 3, 16)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert parameter_count(config) == sum(p.numel() for p in model.parameters())
    assert model.layers[0].attn.rope is None
    if normalization == "none":
        assert not any(isinstance(layer, rmsnorm) for layer in model.modules())


def test_generate_command_reloads_checkpoint_and_tokenizer(tmp_path):
    config = dict(vocab_size=257, context_length=8, d_model=8, num_layers=1, num_heads=2, d_ff=32, rope_theta=10000.0)
    model = TransformerLM(**config)
    optimizer = AdamW(model.parameters())
    save_checkpoint(model, optimizer, 0, tmp_path / "checkpoint.pt")
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with (tmp_path / "vocab.pkl").open("wb") as f:
        pickle.dump({**{i: bytes([i]) for i in range(256)}, 256: b"<|endoftext|>"}, f)
    with (tmp_path / "merges.pkl").open("wb") as f:
        pickle.dump([], f)
    loaded = load_model(tmp_path / "checkpoint.pt")
    torch.testing.assert_close(model(torch.tensor([[1, 2]])), loaded(torch.tensor([[1, 2]])))
    output = tmp_path / "sample.json"
    generate_command(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--vocab",
            str(tmp_path / "vocab.pkl"),
            "--merges",
            str(tmp_path / "merges.pkl"),
            "--prompt",
            "a",
            "--max-new-tokens",
            "2",
            "--temperature",
            "0",
            "--output",
            str(output),
        ]
    )
    assert 1 <= json.loads(output.read_text(encoding="utf-8"))["generated_tokens"] <= 2


def test_official_experiment_plans_preserve_token_budget():
    spec = json.loads((ROOT / "experiments/a1.json").read_text(encoding="utf-8"))
    jobs = make_plan(spec, "batch", "data", "runs", learning_rate=0.001)
    for job in jobs:
        config = job["config"]
        assert config["steps"] * config["batch_size"] * config["context_length"] == 327680000
    ablations = make_plan(spec, "ablation", "data", "runs", learning_rate=0.001)
    variants = {job["name"]: job["config"] for job in ablations}
    assert variants["tinystories-no-norm-low-lr"]["max_lr"] == 0.0001
    assert variants["tinystories-silu"]["d_ff"] == 2048
    assert variants["tinystories-nope"]["no_rope"] is True
    owt = make_plan(spec, "owt", "data", "runs", learning_rate=0.001)[0]["config"]
    assert owt["steps"] == 40000 and owt["vocab_size"] == 32000
    with pytest.raises(ValueError):
        make_plan(spec, "ablation", "data", "runs")


def test_experiment_command_only_plans_by_default(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Planning must not launch a training process")

    monkeypatch.setattr("scripts.run_experiments.subprocess.run", unexpected)
    run_experiments(["--phase", "lr", "--output-root", str(tmp_path / "plan")])
    points, summary = collect_points(tmp_path / "plan")
    assert points == [] and len(summary) == 6
    assert all(row["status"] == "not_run" for row in summary)


def test_preparation_records_hashes_and_rejects_duplicate_splits(tmp_path):
    vocab = tmp_path / "vocab.pkl"
    merges = tmp_path / "merges.pkl"
    with vocab.open("wb") as f:
        pickle.dump({i: bytes([i]) for i in range(256)}, f)
    with merges.open("wb") as f:
        pickle.dump([], f)
    train_text, valid_text = tmp_path / "train.txt", tmp_path / "valid.txt"
    train_text.write_text("Training text.", encoding="utf-8")
    valid_text.write_text("Validation text.", encoding="utf-8")
    output = tmp_path / "tokens"
    common = [
        "--train-text",
        str(train_text),
        "--valid-text",
        str(valid_text),
        "--vocab",
        str(vocab),
        "--merges",
        str(merges),
    ]
    prepare_data(common + ["--output-dir", str(output)])
    manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["sha256"] == file_hash(output / "train.bin")
    assert manifest["splits"]["train"]["tokens"] == len(np.fromfile(output / "train.bin", dtype="<u2"))
    valid_text.write_text(train_text.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="identical"):
        prepare_data(common + ["--output-dir", str(tmp_path / "duplicates")])


def test_accumulated_microbatches_match_a_larger_batch(tmp_path):
    train_path, valid_path = tmp_path / "train.bin", tmp_path / "valid.bin"
    (np.arange(128, dtype=np.uint16) % 16).tofile(train_path)
    (np.arange(64, dtype=np.uint16) % 16).tofile(valid_path)
    common = [
        "--train-data",
        str(train_path),
        "--valid-data",
        str(valid_path),
        "--vocab-size",
        "16",
        "--context-length",
        "4",
        "--d-model",
        "8",
        "--num-layers",
        "1",
        "--num-heads",
        "2",
        "--d-ff",
        "32",
        "--steps",
        "2",
        "--warmup-iters",
        "0",
        "--cosine-cycle-iters",
        "2",
        "--eval-batches",
        "1",
    ]
    full, micro = tmp_path / "full", tmp_path / "micro"
    train(common + ["--batch-size", "2", "--output-dir", str(full)])
    train(common + ["--batch-size", "1", "--gradient-accumulation-steps", "2", "--output-dir", str(micro)])
    a = torch.load(full / "checkpoint.pt", weights_only=True)
    b = torch.load(micro / "checkpoint.pt", weights_only=True)
    torch.testing.assert_close(a["model"], b["model"], rtol=1e-5, atol=1e-6)


def test_resuming_older_checkpoint_archives_superseded_metrics(tmp_path, monkeypatch):
    import importlib

    training = importlib.import_module("cs336_basics.train")
    original_save = training.save_checkpoint

    def save_with_snapshot(model, optimizer, iteration, out, metadata=None):
        original_save(model, optimizer, iteration, out, metadata=metadata)
        if iteration == 2:
            original_save(model, optimizer, iteration, Path(out).parent / "step2.pt", metadata=metadata)

    monkeypatch.setattr(training, "save_checkpoint", save_with_snapshot)
    train_path, valid_path = tmp_path / "train.bin", tmp_path / "valid.bin"
    (np.arange(64, dtype=np.uint16) % 16).tofile(train_path)
    (np.arange(32, dtype=np.uint16) % 16).tofile(valid_path)
    output = tmp_path / "run"
    args = [
        "--train-data",
        str(train_path),
        "--valid-data",
        str(valid_path),
        "--output-dir",
        str(output),
        "--vocab-size",
        "16",
        "--context-length",
        "4",
        "--d-model",
        "8",
        "--num-layers",
        "1",
        "--num-heads",
        "2",
        "--d-ff",
        "32",
        "--batch-size",
        "2",
        "--warmup-iters",
        "0",
        "--cosine-cycle-iters",
        "4",
        "--log-interval",
        "1",
        "--checkpoint-interval",
        "1",
        "--eval-batches",
        "1",
    ]
    train(args + ["--steps", "3"])
    train(args + ["--steps", "4", "--resume", str(output / "step2.pt")])
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2, 3, 4]
    assert [row["elapsed_seconds"] for row in records] == sorted(row["elapsed_seconds"] for row in records)
    backups = list(output.glob("metrics-before-resume-*.jsonl"))
    assert len(backups) == 1
    assert len(backups[0].read_text().splitlines()) == 3
