import pytest
import torch

from cs336_basics.generation import generate
from cs336_basics.transformer_lm import TransformerLM


def make_model(**overrides):
    torch.manual_seed(336)
    config = dict(vocab_size=32, context_length=12, d_model=16, num_layers=2, num_heads=4, d_ff=32, rope_theta=10000.0)
    config.update(overrides)
    return TransformerLM(**config).eval()


@pytest.mark.parametrize("normalization", ["pre", "post", "none"])
@pytest.mark.parametrize("ffn_type", ["swiglu", "silu"])
@pytest.mark.parametrize("use_rope", [True, False])
def test_cached_chunks_match_full_causal_forward(normalization, ffn_type, use_rope):
    model = make_model(normalization=normalization, ffn_type=ffn_type, use_rope=use_rope)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]])
    expected = model(ids)
    cache = None
    chunks = []
    start = 0
    for size in (2, 1, 3):
        logits, cache = model.forward_cached(ids[:, start : start + size], cache)
        chunks.append(logits)
        start += size
        assert len(cache) == 2
        assert all(key.shape == value.shape == (2, 4, start, 4) for key, value in cache)
        assert all(not key.requires_grad and not value.requires_grad for key, value in cache)
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, atol=2e-6, rtol=1e-5)


def test_cached_prefill_does_not_attend_to_future_tokens():
    model = make_model()
    original = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 17, 19]])
    a, _ = model.forward_cached(original)
    b, _ = model.forward_cached(changed)
    torch.testing.assert_close(a[:, :2], b[:, :2], atol=0, rtol=0)


def test_cache_validation_rejects_inconsistent_state_and_context_overflow():
    model = make_model(context_length=4)
    ids = torch.tensor([[1, 2]])
    _, cache = model.forward_cached(ids)
    with pytest.raises(ValueError, match="layer"):
        model.forward_cached(ids[:, :1], cache[:1])
    uneven = [cache[0], (cache[1][0][..., :1, :], cache[1][1][..., :1, :])]
    with pytest.raises(ValueError, match="length"):
        model.forward_cached(ids[:, :1], uneven)
    with pytest.raises(ValueError, match="shape"):
        model.forward_cached(ids.expand(2, -1), cache)
    wrong_dtype = [(key.double(), value.double()) for key, value in cache]
    with pytest.raises(ValueError, match="dtype"):
        model.forward_cached(ids[:, :1], wrong_dtype)
    with pytest.raises(ValueError, match="context_length"):
        model.forward_cached(torch.tensor([[1, 2, 3]]), cache)
    with pytest.raises(ValueError, match="nonempty"):
        model.forward_cached(ids[:, :0])


def test_cached_generation_matches_greedy_decoding_across_window_rebuilds():
    model = make_model(context_length=4)
    model.train()
    prompt = [1, 2, 3]
    plain = generate(model, prompt, 9, temperature=0)
    stats = {}
    cached = generate(model, prompt, 9, temperature=0, use_kv_cache=True, stats=stats)
    assert cached == plain
    assert prompt == [1, 2, 3]
    assert model.training
    assert stats["generated_tokens"] == 9
    assert stats["context_rebuilds"] == 7
    assert stats["cache_tokens"] <= 4
    assert stats["cache_bytes"] == 2 * 2 * 1 * 4 * 4 * 4 * 4
    assert stats["time_to_first_token_seconds"] > 0
    assert stats["decode_seconds"] > 0


def test_cached_generation_has_no_state_between_requests_and_handles_empty_output():
    model = make_model()
    for prompt in ([1, 2], [7, 3, 9], [1, 2]):
        assert generate(model, prompt, 3, temperature=0, use_kv_cache=True) == generate(model, prompt, 3, temperature=0)
    stats = {}
    assert generate(model, [1], 0, use_kv_cache=True, stats=stats) == [1]
    assert stats["generated_tokens"] == 0
    assert stats["time_to_first_token_seconds"] is None
    assert stats["decode_tokens_per_second"] is None


def test_cached_generation_stops_at_eos_and_preserves_state_dict():
    model = make_model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    first = generate(model, [1, 2], 1, temperature=0)[-1]
    stats = {}
    assert generate(model, [1, 2], 10, temperature=0, eos_token_id=first, use_kv_cache=True, stats=stats) == [
        1,
        2,
        first,
    ]
    assert stats["generated_tokens"] == 1
    assert stats["decode_tokens_per_second"] is None
    assert set(model.state_dict()) == set(before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_inference_benchmark_records_real_measurements_and_rejects_overwrite(tmp_path):
    import json
    import pickle

    from cs336_basics.optimizer import AdamW
    from cs336_basics.training_utils import save_checkpoint
    from scripts.benchmark_inference import main

    model = make_model(vocab_size=257, context_length=8)
    config = dict(vocab_size=257, context_length=8, d_model=16, num_layers=2, num_heads=4, d_ff=32, rope_theta=10000.0)
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_checkpoint(model, AdamW(model.parameters()), 1, tmp_path / "checkpoint.pt")
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    for name, content in (("vocab.pkl", vocab), ("merges.pkl", [])):
        with (tmp_path / name).open("wb") as target:
            pickle.dump(content, target)
    (tmp_path / "prompt.txt").write_text("abcdefghijk")
    arguments = [
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--vocab",
        str(tmp_path / "vocab.pkl"),
        "--merges",
        str(tmp_path / "merges.pkl"),
        "--text-file",
        str(tmp_path / "prompt.txt"),
        "--prompt-lengths",
        "2",
        "4",
        "--new-tokens",
        "3",
        "--repeats",
        "2",
        "--warmup",
        "0",
        "--threads",
        "1",
        "--output",
        str(tmp_path / "benchmark.json"),
    ]
    report = main(arguments)
    assert report["all_greedy_ids_match"]
    assert len(report["samples"]) == 8 and len(report["summary"]) == 2
    assert report["maximum_logit_absolute_error"] < 1e-4
    assert all(row["decode_tokens_per_second"] > 0 for row in report["samples"])
    with pytest.raises(FileExistsError):
        main(arguments)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"name": "cross-window", "text": "abcdefghijk", "prompt_tokens": 7, "new_tokens": 3}]))
    case_args = arguments.copy()
    index = case_args.index("--text-file")
    case_args[index : index + 2] = ["--cases-file", str(cases)]
    case_args[-1] = str(tmp_path / "boundary.json")
    with pytest.raises(ValueError, match="context rebuilds"):
        main(case_args)
    report = main([*case_args, "--allow-context-rebuilds"])
    assert report["all_greedy_ids_match"] and report["allow_context_rebuilds"]
    assert report["summary"][0]["cached"]["context_rebuilds"] == 1
    assert report["summary"][0]["uncached"]["context_rebuilds"] == 0
