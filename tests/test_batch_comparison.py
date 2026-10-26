import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_experiments import make_plan


@pytest.fixture
def small_training(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    train = tmp_path / "train.bin"
    valid = tmp_path / "valid.bin"
    (np.arange(128, dtype=np.uint16) % 16).tofile(train)
    (np.arange(64, dtype=np.uint16)[::-1] % 16).tofile(valid)
    args = [
        "--train-data", str(train), "--valid-data", str(valid), "--vocab-size", "16",
        "--context-length", "4", "--d-model", "8", "--num-layers", "1", "--num-heads", "2",
        "--d-ff", "32", "--steps", "2", "--warmup-iters", "0", "--cosine-cycle-iters", "2",
        "--eval-interval", "2", "--eval-batches", "2", "--device", "cpu",
    ]
    yield args
    torch.set_num_threads(previous)


def test_batch_sweep_uses_same_validation_examples_and_token_cadence():
    spec = json.loads((Path(__file__).resolve().parents[1] / "experiments/a1.json").read_text())
    jobs = make_plan(spec, "batch", "data", "runs", learning_rate=0.001, batch_sizes=[1, 8, 32, 64, 128, 256])
    for job in jobs:
        config = job["config"]
        assert config["eval_batch_size"] == 32 and config["eval_batches"] == 20
        assert config["steps"] * job["effective_batch_size"] * config["context_length"] == 327680000
        for field in ("log_interval", "eval_interval", "checkpoint_interval"):
            observed = config[field] * job["effective_batch_size"]
            expected = spec["baseline"][field] * spec["baseline"]["batch_size"]
            assert abs(observed - expected) <= job["effective_batch_size"] / 2


def test_training_batch_does_not_change_fixed_validation_inputs(tmp_path, small_training, monkeypatch):
    training = importlib.import_module("cs336_basics.train")
    original = training.get_batch
    observed = []

    def capture(*args, **kwargs):
        x, y = original(*args, **kwargs)
        if kwargs.get("rng") is not None:
            observed.append(x.clone())
        return x, y

    monkeypatch.setattr(training, "get_batch", capture)
    samples = []
    for batch in (1, 8):
        observed.clear()
        training.main(small_training + ["--batch-size", str(batch), "--eval-batch-size", "4",
                                       "--output-dir", str(tmp_path / f"batch-{batch}")])
        samples.append(torch.cat(observed))
    assert samples[0].shape == samples[1].shape == (8, 4)
    torch.testing.assert_close(samples[0], samples[1], rtol=0, atol=0)


def test_checkpoint_without_eval_batch_option_remains_resumable(tmp_path, small_training):
    training = importlib.import_module("cs336_basics.train")
    output = tmp_path / "legacy"
    args = small_training + ["--batch-size", "2", "--output-dir", str(output)]
    training.main(args + ["--steps", "1"])
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("eval_batch_size", None)
    config_path.write_text(json.dumps(config))
    training.main(args + ["--resume", str(output / "checkpoint.pt")])
    assert torch.load(output / "checkpoint.pt", weights_only=True)["iteration"] == 2
