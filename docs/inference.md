# KV cache inference

## Implementation

Uncached generation recomputes all context positions at every token. `TransformerLM.forward_cached` returns logits and a `(K, V)` pair per layer. Prefill processes the prompt; subsequent passes process only the most recently generated token and reuse previous keys and values.

Cache state is request-local, passed explicitly, with shape `(...batch, heads, cached_tokens, head_dim)` per tensor. It adds no parameters or checkpoint keys. Keys are cached after RoPE. New query/key positions start at the cached length. A rectangular causal mask allows position `p` to attend only to keys `<= p`, including when new input contains multiple tokens.

`generate(..., use_kv_cache=True)` or `scripts.generate --kv-cache` opts in. The uncached path remains available. Generation restores training/evaluation state, handles EOS, and retains no cache across requests. Layer count, shape, dtype/device and context length are checked.

## Window semantics

The existing generator crops to the last `context_length` tokens and recomputes at positions starting from zero. Simply dropping the oldest K/V entry is not equivalent: retained hidden states already attended to the removed prefix. To preserve behavior, the generator **rebuilds the entire window when it fills**.

Long generation at a full window triggers repeated rebuilds and loses the acceleration. Cache tensors use dynamic concatenation rather than preallocated/paged storage. This is single-process inference, not concurrent serving or a long-context solution.

## Real checkpoint benchmark

Measured 2026-10-26 on i9-13900HX, WSL2 Linux, PyTorch 2.11.0+cu130, CPU FP32, four threads, batch 1. The actual 22,696,448-parameter TinyStories batch-64 checkpoint is loaded; loading/tokenization are outside timing.

Each case uses the first 16, 64 or 128 tokens of the [recorded story](../benchmarks/tinystories/sample.txt), then generates 64 greedy tokens. EOS is disabled for equal budgets. Both paths receive one warmup; order alternates over three paired measured repeats. Prefixes of one generated story do not represent a diverse workload or quality evaluation.

| Prompt tokens | Uncached decode tokens/s | Cached decode tokens/s | Speedup | First token ms, uncached / cached | K/V MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 143.65 | 391.94 | 2.73× | 4.08 / 4.19 | 1.234 |
| 64 | 89.38 | 374.61 | 4.19× | 9.30 / 9.54 | 1.984 |
| 128 | 58.36 | 364.05 | 6.24× | 13.69 / 15.25 | 2.984 |

Each speed is a three-repeat median; [all raw measurements](../benchmarks/kv-cache/cpu-20261026.json) are included. Decode throughput excludes the first token: 63 tokens divided by remaining generation time. First-token latency includes prefill and selection, and did not improve. No window rebuild occurs here. CPU scheduling/frequency and other workloads affect short measurements; there is no confidence interval or GPU speedup claim.

Greedy IDs match between paths in all measured cases. Separately, chunked cached logits match full logits at `atol=rtol=1e-4`; maximum absolute error is **1.1444091796875e-5**. Bitwise equality is not required. Near-ties could change greedy selection for other inputs, so measured equality is not a universal guarantee.

FP32 K/V tensor storage is `2 * layers * batch * cached_length * d_model * 4` bytes. The largest case ends with cached length 191 (128 prompt + 63 consumed generated tokens), or 3,129,344 bytes. This excludes process peak RAM, temporary allocations and model memory.

## Reproduce

Provide a trained checkpoint/config and matching tokenizer locally; weights are not in Git. Measured checkpoint SHA-256: `c41dbd96c25aa836dbd13c58fd3113fd6acbd3b2d09699e6e51878d699739789`. Tokenizer hashes are in the JSON.

```bash
uv run --frozen python -m scripts.benchmark_inference \
  --checkpoint runs/a1-batch/tinystories-batch-64/checkpoint.pt \
  --vocab results/tinystories-train/vocab.pkl \
  --merges results/tinystories-train/merges.pkl \
  --text-file benchmarks/tinystories/sample.txt \
  --prompt-lengths 16 64 128 --new-tokens 64 \
  --warmup 1 --repeats 3 --threads 4 --device cpu \
  --output results/kv-cache-cpu.json
```

The command rejects existing output and stops on correctness mismatch. CUDA mode supports synchronized timing but requires a separate actual measurement. Inference caching does not alter training.

## Multi-prompt and window follow-up

A second run uses the inference-only export in a fresh extracted source directory, with the same CPU/thread/precision settings and three alternating paired repeats. Inputs/token budgets were fixed in [cases-20261026.json](../benchmarks/kv-cache/cases-20261026.json). All 30 measurements and per-case token IDs are in [cpu-boundaries-20261026.json](../benchmarks/kv-cache/cpu-boundaries-20261026.json); original results above remain a separate experiment.

| Case | Prompt / new tokens | Uncached / cached decode tokens/s | Speedup | Rebuilds |
| --- | ---: | ---: | ---: | ---: |
| Story | 16 / 64 | 144.22 / 381.98 | 2.65× | 0 |
| Engineering text | 64 / 64 | 85.19 / 368.68 | 4.33× | 0 |
| Multilingual text | 128 / 64 | 59.01 / 371.93 | 6.30× | 0 |
| Near context limit | 240 / 16 | 39.30 / 346.52 | 8.82× | 0 |
| Cross context limit | 248 / 32 | 37.01 / 45.59 | 1.23× | 23 |

Each pair has equal prompt/output budgets. Different workloads/output lengths are not pooled into an overall speedup. The near-window case stays within the cache; it does not establish an 8.82× long-generation improvement. The crossing case includes 23 full rebuilds and loses most of the acceleration. Starting generation with a full window requires rebuilding at each subsequent step, so no universal full-window speedup is claimed.

All greedy IDs match. Chunked-prefill logits meet `atol=rtol=1e-4`, maximum absolute error **1.5735626220703125e-5**. Records retain TTFT,total elapsed,decode time,cache bytes and rebuilds; first-token latency has no consistent improvement. EOS is disabled during timing only. Separately,the actual saved finished story with EOS removed emits real EOS ID256 as its first new token in both modes and stops correctly. Five synthetic/repeated-text inputs and three short repeats remain a limited workload,not a quality or production-serving benchmark.

See [delivery and reproduction](delivery.md) for artifact identities and commands. Existing dependencies were reused; no fresh-machine installation or GPU measurement is claimed.
