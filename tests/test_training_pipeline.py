import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cs336_basics.optimizer import AdamW, SGD
from cs336_basics.train import load_tokens, main
from cs336_basics.training_utils import cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_cross_entropy_batch_dimensions_and_gradients():
    inputs = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0, 1, 2], [3, 4, 0]])
    actual = cross_entropy(inputs, targets)
    expected = F.cross_entropy(inputs.reshape(-1, 5), targets.reshape(-1))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, inputs, retain_graph=True)[0],
        torch.autograd.grad(expected, inputs)[0],
    )
    torch.testing.assert_close(cross_entropy(inputs + 1000, targets), expected)
    half_inputs = torch.tensor([[60000.0, -60000.0]], dtype=torch.float16)
    assert torch.isfinite(cross_entropy(half_inputs, torch.tensor([1])))


def test_batch_shortest_valid_dataset():
    dataset = np.arange(5, dtype=np.uint16)
    x, y = get_batch(dataset, 2, 4, "cpu")
    torch.testing.assert_close(x, torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))
    torch.testing.assert_close(y, torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]))
    with pytest.raises(ValueError):
        get_batch(dataset, 2, 5, "cpu")


def test_clipping_accepts_generator_and_skips_missing_gradient():
    parameter = torch.nn.Parameter(torch.ones(2))
    missing = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    gradient_clipping(iter([parameter, missing]), 1.0)
    torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 4.0]) / (5.0 + 1e-6))
    assert missing.grad is None


def test_schedule_without_warmup_and_after_cycle():
    assert get_lr_cosine_schedule(0, 1.0, 0.1, 0, 4) == 1.0
    assert get_lr_cosine_schedule(4, 1.0, 0.1, 0, 4) == 0.1
    assert get_lr_cosine_schedule(5, 1.0, 0.1, 0, 4) == 0.1
    with pytest.raises(ValueError):
        get_lr_cosine_schedule(0, 1.0, 0.1, 4, 4)


def test_adamw_first_step_and_missing_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    missing = torch.nn.Parameter(torch.tensor([3.0]))
    parameter.grad = torch.tensor([0.3, -0.4])
    optimizer = AdamW([parameter, missing], lr=0.01, betas=(0.9, 0.95), weight_decay=0.1)
    m = 0.1 * parameter.grad
    v = 0.05 * parameter.grad.square()
    adjusted_lr = 0.01 * math.sqrt(0.05) / 0.1
    expected = parameter.detach() * 0.999 - adjusted_lr * m / (v.sqrt() + 1e-8)
    optimizer.step()
    torch.testing.assert_close(parameter, expected)
    torch.testing.assert_close(missing, torch.tensor([3.0]))
    assert missing not in optimizer.state


def test_sgd_uses_handout_decay():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.ones(1)
    optimizer = SGD([parameter], lr=1.0)
    optimizer.step()
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([-1 / math.sqrt(2)]))


def test_memory_mapping_and_invalid_token_ids(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(16, dtype=np.uint16))
    assert isinstance(load_tokens(path, "uint16", 16, 4), np.memmap)
    with pytest.raises(ValueError, match="Token IDs"):
        load_tokens(path, "uint16", 8, 4)


def test_training_resume_matches_uninterrupted(tmp_path):
    train_path = tmp_path / "train.bin"
    valid_path = tmp_path / "valid.bin"
    (np.arange(128, dtype=np.uint16) % 16).tofile(train_path)
    (np.arange(64, dtype=np.uint16)[::-1] % 16).tofile(valid_path)
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
        "64",
        "--batch-size",
        "2",
        "--warmup-iters",
        "1",
        "--cosine-cycle-iters",
        "4",
        "--log-interval",
        "1",
        "--eval-interval",
        "3",
        "--eval-batches",
        "1",
        "--checkpoint-interval",
        "2",
        "--device",
        "cpu",
    ]
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"
    main(common + ["--steps", "4", "--output-dir", str(full_dir)])
    main(common + ["--steps", "2", "--output-dir", str(resumed_dir)])
    main(common + ["--steps", "4", "--output-dir", str(resumed_dir), "--resume", str(resumed_dir / "checkpoint.pt")])

    full = torch.load(full_dir / "checkpoint.pt", weights_only=True)
    resumed = torch.load(resumed_dir / "checkpoint.pt", weights_only=True)
    assert full["iteration"] == resumed["iteration"] == 4
    torch.testing.assert_close(full["model"], resumed["model"], rtol=0, atol=0)
    torch.testing.assert_close(full["optimizer"], resumed["optimizer"], rtol=0, atol=0)
    torch.testing.assert_close(full["torch_rng_state"], resumed["torch_rng_state"], rtol=0, atol=0)
    assert full["numpy_rng_state"] == resumed["numpy_rng_state"]
    records = [json.loads(line) for line in (resumed_dir / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2, 3, 4]
    assert all(math.isfinite(record["train_loss"]) for record in records)
    assert "validation_loss" in records[-1]
    assert records[-1]["elapsed_seconds"] > records[1]["elapsed_seconds"]

    with pytest.raises(FileExistsError):
        main(common + ["--steps", "4", "--output-dir", str(full_dir)])
    with pytest.raises(ValueError, match="separate files"):
        main(common + ["--train-data", str(valid_path), "--output-dir", str(tmp_path / "leak")])
