# Project scope and remaining experiments

Specification: Stanford CS336 A1, commit `a158843b20107949f1a8d7df1b05cd33b9166712`, handout 26.0.3. Reviewed 2026-10-26.

This project implements a TinyStories language-model training and inference pipeline. It includes measured engineering extensions beyond the assignment, while the full OpenWebText workload and several written experiment deliverables remain outstanding.

## Implemented and measured

| Area | Available evidence |
| --- | --- |
| Byte-level BPE and tokenizer | Upstream tests, incremental-count equivalence, round-trip and streaming checks |
| TinyStories BPE | Full 10K-vocabulary run with training time, peak RAM and longest token |
| Data preparation | Separate training/validation encoding, token counts and source hashes |
| Transformer / language model | Component snapshots, causal attention/RoPE checks and end-to-end tests |
| Training utilities | Loss, AdamW, schedule, clipping, data sampling and checkpoint/resume tests |
| Resource accounting / SGD | [Analytical calculations and four CPU runs](accounting.md) |
| Learning-rate sweep | Five completed candidates and one nonfinite-gradient failure |
| Batch comparison | Batches 1/8/32/64/128/256 complete; 320 OOM; equal token budgets and common validation inputs |
| Architecture ablations | Five completed runs with a measured batch-32 control |
| Text generation | Actual checkpoint sample: 150 new tokens, then EOS |
| KV cache inference | Request-local state, RoPE offsets, rectangular masks and exact context-window rebuilds |
| Inference delivery | Parameter-identical export, fresh-directory CPU generation, natural EOS and 30 paired workload measurements |

See [experiment results](results.md), [inference methodology](inference.md) and [artifact reproduction](delivery.md). Local Linux CPU validation is 103 passed and 1 xpassed; the manual GitHub workflow has not been run remotely.

## Remaining official A1 work

- OpenWebText BPE training, tokenizer encoding and full language-model experiment, including held-out curves, generation and comparison.
- The prescribed ten-document and cross-corpus tokenizer measurements, extrapolation answers and profiler-based BPE analysis.
- Complete problem-by-problem written answers alongside the existing numerical results and accounting.
- Conditional learning-rate retuning for small batches before drawing stronger batch-size conclusions.

The optional leaderboard has not been attempted. No result under its hardware/time rules is claimed.

## Interpretation limits

Model selection uses sampled validation loss and a single seed. The batch sweep keeps peak learning rate fixed; it does not establish the best attainable model at each batch size. More seeds and an untouched test split would be needed for stronger generalization claims.

Inference speedups are CPU decode measurements on fixed workloads, not first-token, GPU or concurrent-serving results. KV caching is a standard technique. Full-window reconstruction preserves the original cropped-window semantics but reduces its benefit after the context fills. Log truncation was handled by separate, traceable recovery from intact console records; its cause remains unknown.
