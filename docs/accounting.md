# Resource accounting

The assignment model has untied input/output embeddings, bias-free projections, two RMSNorms per block, one final RMSNorm and a three-matrix SwiGLU FFN. Let `V,S,L,d,f,H,B` denote vocabulary size, context length, layers, model width, FFN width, heads and batch size. A multiply-add counts as two FLOPs. GB is decimal; GiB is binary.

## Transformer parameters and forward FLOPs

```text
P = 2 V d + L (4 d² + 3 d f + 2 d) + d
```

These terms count embeddings/head, attention projections, FFN weights, block normalization scales and final normalization scales. RoPE has no trainable parameters.

GPT-2 XL shape (`V=50257,S=1024,L=48,d=1600,H=25,f=4288`) has **1,640,452,800 parameters**. FP32 weights alone occupy **6,561,811,200 bytes = 6.562 GB ≈ 6.111 GiB**, excluding gradients, moments and activations.

| Forward matrix operation | FLOPs per layer, one sequence |
| --- | ---: |
| Three Q/K/V projections | `6 S d²` |
| QK scores, all heads | `2 S² d` |
| Probability-weighted values | `2 S² d` |
| Attention output projection | `2 S d²` |
| Three SwiGLU projections | `6 S d f` |
| Vocabulary head (once, outside layers) | `2 S d V` |

```text
F(S) = L (8 S d² + 4 S² d + 6 S d f) + 2 S d V
```

This matrix-only count omits norm, softmax, RoPE and activation scalar operations. Embeddings are lookups. Dense attention counts full matrices despite the causal mask. XL at context 1024 requires **3,516,769,894,400 FLOPs** per sequence (3.517 TFLOPs). FFN multiplies dominate, followed by attention projections.

With `f` rounded to the nearest multiple of 64 to `8d/3`, and `S=1024`:

| Shape (L,d,H,f) | Forward TFLOPs | Attention projections | QK + weighted V | FFN | Vocabulary head |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small (12,768,12,2048) | 0.29165 | 19.881% | 13.254% | 39.762% | 27.104% |
| Medium (24,1024,16,2752) | 0.83017 | 24.833% | 12.417% | 50.054% | 12.696% |
| Large (36,1280,20,3392) | 1.76853 | 27.321% | 10.928% | 54.301% | 7.449% |
| XL (48,1600,25,4288) | 3.51677 | 28.624% | 9.160% | 57.534% | 4.683% |

At fixed context/vocabulary, increasing width/depth increases the relative FFN/projection costs while the one-time vocabulary head takes a smaller share. At context 16384, XL requires **133,577,729,638,400 FLOPs**, about 133.578 TFLOPs or **37.983×**, rather than 16×. QK/weighted-V grows quadratically; other listed multiplies grow linearly. Attention's quadratic portion rises to about 61.74% of the total from 9.16%.

## AdamW memory and compute

For this question use `f=8d/3` exactly, as specified, rather than rounding:

```text
P = 2 V d + L (12 d² + 2 d) + d
parameters:       4P bytes
gradients:        4P bytes
Adam moments m,v: 8P bytes
```

An explicit simplified activation ledger counts the outputs of the listed operations, retained in FP32. Each block has norm outputs `2BSd`, QKV `3BSd`, weighted-V/output projection `2BSd`, FFN output `BSd`, W1/W3/SiLU/product `4BSf`, and scores/probabilities `2BHS²`. Add final norm `BSd`, logits `BSV` and cross-entropy probabilities `BSV`:

```text
A = B S [L ((56/3) d + 2 H S) + d + 2 V] floats
total estimated bytes = 16P + 4A
```

For idealized XL, `P=1,635,537,600`:

```text
estimated bytes = 26,168,601,600 + 16,356,614,144 B
```

This ledger gives a maximum **B=3 within decimal 80 GB**. It is an analytical estimate, not a measured safe batch size. Runtime/allocator overhead, residual/workspace storage and some backward intermediates are excluded. Autograd reuse, fused attention, checkpointing and mixed precision change actual peak memory; profile the implementation before claiming fit.

One AdamW update is `Θ(P)` scalar operations. A convention with precomputed group bias factors counts approximately **14P**: 3 for first moments, 4 for second moments, 5 for normalized updates and 2 for decay. Square roots/divisions count as one operation each; kernel fusion/operation conventions change the constant. Matrix FLOPs dominate for these sequence lengths.

Using rounded XL forward FLOPs, backward at twice forward, batch 1024, 400,000 updates and achieved `0.5 * 495e12` FLOP/s:

```text
hours = 400000 * (3 * 1024 * F(1024) + 14P) / (0.5 * 495e12) / 3600
      ≈ 4850.1 hours, or 202.1 days
```

This follows the handout's hypothetical H100/MFU assumptions. It neither establishes batch-1024 memory fit nor includes data, communication or checkpoint overhead. Rounding the FFN width explains the small parameter-count difference between the two questions.

## SGD learning-rate observation

[Recorded CPU results](../benchmarks/sgd) use the same seed-42 10×10 initial matrix, `mean(W²)` loss, step size `lr/sqrt(t+1)` and ten updates. Loss is logged before each update.

| Initial LR | First loss | Last logged loss (step 9) |
| ---: | ---: | ---: |
| 1 | 24.169258 | 20.000328 |
| 10 | 24.169258 | 3.248231 |
| 100 | 24.169258 | 2.3653e-23 |
| 1000 | 24.169258 | 2.2402e18 |

Here `grad=2W/100`, hence `W[t+1]=(1-lr/(50 sqrt(t+1)))W[t]`. LR 1 and 10 contract at different speeds. LR 100 initially flips signs at equal magnitude, then contracts rapidly as its step decays. LR 1000 amplifies weights severely within this window. This is a quadratic exercise, not a transferable Transformer LR scale; the last log is before the final update.

The following standard-library check reproduces the main XL numbers:

```python
V, S, L, d, f, H = 50257, 1024, 48, 1600, 4288, 25
P = 2*V*d + L*(4*d*d + 3*d*f + 2*d) + d
F = lambda s: L*(8*s*d*d + 4*s*s*d + 6*s*d*f) + 2*s*d*V
assert (P, 4*P, F(S), F(16384)) == (
    1640452800, 6561811200, 3516769894400, 133577729638400)
```
