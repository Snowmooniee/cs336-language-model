import json
import pickle

import pytest
import torch

from cs336_basics.generation import generate, load_model
from cs336_basics.optimizer import AdamW
from cs336_basics.training_utils import save_checkpoint
from cs336_basics.transformer_lm import TransformerLM
from scripts.export_inference import main
from scripts.prepare_data import file_hash


def test_inference_export_strips_training_state_and_preserves_parameters(tmp_path):
    config = dict(vocab_size=257, context_length=8, d_model=16, num_layers=1, num_heads=4, d_ff=32, rope_theta=10000.0)
    model = TransformerLM(**config)
    optimizer = AdamW(model.parameters())
    model(torch.tensor([[1, 2]])).sum().backward()
    optimizer.step()
    save_checkpoint(model, optimizer, 1, tmp_path / "checkpoint.pt")
    (tmp_path / "config.json").write_text(
        json.dumps({**config, "train_data": "private-path", "api_key": "not-for-export"})
    )
    vocab = {index: bytes([index]) for index in range(256)}
    vocab[256] = b"<|endoftext|>"
    for name, value in (("vocab.pkl", vocab), ("merges.pkl", [])):
        with (tmp_path / name).open("wb") as target:
            pickle.dump(value, target)
    args = [
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--vocab",
        str(tmp_path / "vocab.pkl"),
        "--merges",
        str(tmp_path / "merges.pkl"),
        "--source-revision",
        "test-revision",
        "--expected-checkpoint-sha256",
        file_hash(tmp_path / "checkpoint.pt"),
        "--output-dir",
        str(tmp_path / "inference"),
    ]
    manifest = main(args)
    output = tmp_path / "inference"
    assert manifest["source_iteration"] == 1
    assert manifest["source_checkpoint_sha256"] != manifest["files"]["weights.pt"]["sha256"]
    assert manifest["all_exported_parameters_equal"]
    weights = torch.load(output / "weights.pt", weights_only=True)
    assert set(weights) == {"model"}
    exported_config = json.loads((output / "config.json").read_text())
    assert "api_key" not in exported_config and "train_data" not in exported_config
    restored = load_model(output / "weights.pt")
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[name])
    assert generate(restored, [1, 2], 4, temperature=0, use_kv_cache=True) == generate(model, [1, 2], 4, temperature=0)
    for line in (output / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ")
        assert file_hash(output / name) == checksum
    assert {path.name for path in output.iterdir()} == {
        "weights.pt",
        "config.json",
        "vocab.pkl",
        "merges.pkl",
        "manifest.json",
        "SHA256SUMS",
    }
    with pytest.raises(FileExistsError):
        main(args)


def test_export_rejects_bad_source_hash_before_writing(tmp_path):
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        main(
            [
                "--checkpoint",
                str(source),
                "--vocab",
                "unused",
                "--merges",
                "unused",
                "--source-revision",
                "test",
                "--expected-checkpoint-sha256",
                "0" * 64,
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert not (tmp_path / "output").exists()
