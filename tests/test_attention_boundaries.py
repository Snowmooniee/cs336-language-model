"""Supplementary attention checks beyond the official assignment tests."""

import torch

from .adapters import run_scaled_dot_product_attention


def test_masked_key_stays_excluded_for_large_negative_scores():
    q = torch.tensor([[[1.0]]])
    k = torch.tensor([[[-2e9], [0.0]]])
    v = torch.tensor([[[7.0], [123.0]]])
    mask = torch.tensor([[True, False]])

    actual = run_scaled_dot_product_attention(q, k, v, mask)

    torch.testing.assert_close(actual, torch.tensor([[[7.0]]]), rtol=0, atol=0)


def test_masked_attention_supports_float16():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float16)
    v = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]], dtype=torch.float16)
    mask = torch.eye(2, dtype=torch.bool)

    actual = run_scaled_dot_product_attention(q, q, v, mask)

    assert actual.dtype == torch.float16
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, v, rtol=0, atol=0)
