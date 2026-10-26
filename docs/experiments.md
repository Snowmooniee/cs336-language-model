# Training, decoding and experiments

Specification: Stanford CS336 A1 handout 26.0.3, sections 6–7. The experiment definition is `experiments/a1.json`.

The commands below prepare or execute experiments; their presence does not mean that a run has completed. Large run outputs are ignored. Measured TinyStories results are published under `benchmarks/tinystories`; see [analysis](results.md), [inference extension](inference.md) and [remaining assignment work](assignment-status.md).

## Resource checks

System RAM, WSL memory and GPU VRAM are separate limits. WSL does not add physical RAM to the Windows host. Full BPE training currently reads the input into memory and holds pre-token and merge statistics, so preprocessing needs a separate RAM assessment.

The default baseline has vocabulary 10000, context 256, width 512, FFN width 1344, 4 layers and 16 heads. It contains 22,696,448 parameters. FP32 weights occupy about 86.58 MiB; weights, gradients and two Adam moment tensors occupy about 346.32 MiB. These figures exclude activations, temporary tensors, runtime libraries, data and allocator overhead; they cannot establish a safe batch size.

Static estimate, without importing PyTorch or creating a model:

```bash
uv run python -m scripts.benchmark --output results/model-estimate.json
```

On a machine already selected for training, explicitly measure synthetic forward/backward/optimizer steps:

```bash
uv run python -m scripts.benchmark --execute \
  --device cuda --batch-sizes 1 4 8 16 32 64 128 \
  --output results/gpu-benchmark.json
```

The benchmark stops at the first OOM. It reports allocated/reserved VRAM and measured tokens/s; its runtime estimate excludes preprocessing, validation and checkpoint overhead. It does not provision machines or submit paid jobs.

## Prepare separate data splits

Obtain the official TinyStories and OpenWebText text files from the [pinned upstream setup instructions](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/README.md#download-data). Training and validation must be genuinely separate splits.

Train the vocabularies with `scripts.train_bpe`: TinyStories uses 10000 entries and OpenWebText 32000. For example:

```bash
uv run python -m scripts.train_bpe \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output-dir results/tinystories-train --vocab-size 10000
```

Encode one dataset at a time using its own trained tokenizer:

```bash
uv run python -m scripts.prepare_data \
  --train-text data/TinyStoriesV2-GPT4-train.txt \
  --valid-text data/TinyStoriesV2-GPT4-valid.txt \
  --vocab results/tinystories-train/vocab.pkl \
  --merges results/tinystories-train/merges.pkl \
  --output-dir data/tokens/tinystories
```

Repeat for OpenWebText, writing to `data/tokens/owt`. Preparation writes `train.bin`, `valid.bin` and `dataset.json` with token counts and SHA-256 fingerprints. It rejects identical input splits and existing output directories. Each split is encoded as a little-endian uint16 stream.

The encoder streams text, but the BPE trainer is not yet a streaming trainer. Sufficient preprocessing RAM and measured preprocessing time remain necessary.

## SGD learning-rate observations

This is the handout's `learning_rate_tuning` exercise, separate from the Transformer sweep:

```bash
uv run python -m scripts.sgd_experiment --output-dir runs/sgd
```

The default rates are 1, 10, 100, 1000, using the same initial 10×10 weight matrix and 10 updates. Logged losses are measured before each update. Write the requested observations from the resulting curves rather than assuming that a larger rate always helps.

## Plan the full Transformer experiments

Every default run processes 327,680,000 tokens. With physical batch 32 and context 256, that is 40000 optimizer updates. The full target validation loss remains 1.45 for TinyStories; no reduced-resource target is silently substituted.

Plan a learning-rate sweep without launching training:

```bash
uv run python -m scripts.run_experiments \
  --phase lr --data-root data/tokens --output-root runs/a1-lr
```

Inspect `plan.json`. Add `--execute` only on the intended compute machine. `--max-runs` bounds the number of jobs, and `--max-seconds` bounds each training subprocess. Re-running a suite skips completed jobs, but does not overwrite a failed job directory.

An explicitly reduced 100-update infrastructure check, not a full assignment experiment:

```bash
uv run python -m scripts.run_experiments \
  --phase lr --data-root data/tokens --output-root runs/preflight \
  --total-tokens 819200 --max-runs 1 --max-seconds 900 --execute
```

Choose subsequent rates from measured curves. The published TinyStories sweep selected peak LR 0.001; LR 0.1 stopped with a nonfinite gradient. This selection is specific to the measured configuration and does not establish an OWT optimum:

```bash
uv run python -m scripts.run_experiments \
  --phase baseline --learning-rate 0.001 \
  --data-root data/tokens --output-root runs/a1-baseline

uv run python -m scripts.run_experiments \
  --phase batch --learning-rate 0.001 \
  --data-root data/tokens --output-root runs/a1-batch

uv run python -m scripts.run_experiments \
  --phase ablation --learning-rate 0.001 \
  --data-root data/tokens --output-root runs/a1-ablation

uv run python -m scripts.run_experiments \
  --phase owt --learning-rate 0.001 \
  --data-root data/tokens --output-root runs/a1-owt
```

These commands still only plan; add `--execute` deliberately. Baseline, batch, ablation and OWT phases require an explicit learning rate. The suite records the source fingerprint, configuration, physical/effective batch sizes, token budget, completion status and actual metric files. Failed jobs are distinguished from OOM, nonfinite divergence and timeouts.

The batch phase keeps processed tokens fixed while changing physical batch size across 1, 8, 32, 64, 128. OOM is an observed resource result. If rates need retuning for a batch size, rerun that size with a new output root and the selected rate.

Batch-phase validation uses the baseline batch size (32) and the same 20 validation batches with a fixed sampling seed, regardless of the training batch size. This evaluates the same 640 held-out sequences for each candidate. Logging, validation and checkpoint intervals are scaled inversely with effective training batch size to keep their token spacing approximately constant; rounding to an integer update can shift a boundary by at most half an update. The default training CLI still evaluates at its training batch size unless `--eval-batch-size` is provided, and older checkpoints remain resumable.

Use the resource probe to extend the batch range to the hardware limit rather than assuming 128 is the maximum. For example, probe 256 and 512 before choosing the largest measured fitting size. Run intermediate learning curves at equal training-token budgets; a synthetic throughput check alone does not satisfy the batch-size experiment. A smaller batch can require many more optimizer updates and much longer wall time for that same token budget.

`--micro-batch-size` permits explicit gradient accumulation. For example, effective batch 32 with microbatch 4 performs 8 backward passes before an optimizer update. Such runs are labelled in the plan and name; they do not demonstrate that physical batch 32 fit in memory.

## Ablations

| Run | Difference from baseline |
| --- | --- |
| baseline | Pre-norm, RoPE, SwiGLU |
| no-norm | All RMSNorms removed, including final RMSNorm |
| no-norm-low-lr | Same removal, chosen rate divided by10 |
| post-norm | Block normalization follows each residual addition |
| nope | No RoPE on queries or keys |
| silu | Two-matrix SiLU FFN with width 2048 instead of three-matrix SwiGLU width 1344 |

The SiLU widths approximately match FFN parameter counts. Baseline defaults remain unchanged for upstream snapshot tests. All architecture choices are serialized in the run config, so loading a checkpoint restores its matching architecture.

OWT keeps the baseline Transformer body, context length and update count; its own tokenizer uses vocabulary 32000. This changes the embedding/head parameter counts. Per-token losses across different tokenizers should not be treated as directly equivalent measures of corpus difficulty or text quality.

## Decode from a checkpoint

```bash
uv run python -m scripts.generate \
  --checkpoint runs/a1-baseline/tinystories-baseline/checkpoint.pt \
  --vocab results/tinystories-train/vocab.pkl \
  --merges results/tinystories-train/merges.pkl \
  --prompt "Once upon a time" \
  --max-new-tokens 256 --temperature 0.8 --top-p 0.9 \
  --output results/tinystories-sample.json
```

The default device is CPU. Temperature 0 uses greedy decoding. Top-p keeps the smallest probability prefix reaching the threshold, including the crossing token. Generation stops at EOS or the token limit and crops input to context length. Add `--kv-cache` for cached decoding; reaching the context limit rebuilds the cropped window to preserve existing semantics. The saved JSON also records timing and retained cache bytes.

The saved JSON includes the generated IDs, count, EOS-stop status and decoded text. For the handout, report at least 256 generated tokens or an earlier EOS, then discuss fluency and at least two factors affecting output quality. Code availability or an untrained sample cannot satisfy that requirement.

## Curves and experiment log

Create CSV summaries from actual recorded jobs:

```bash
uv run python -m scripts.report_experiments \
  --run-root runs/a1-lr --output-dir results/a1-lr-report
```

For plots, Matplotlib is an optional reporting dependency; the core lockfile is unchanged:

```bash
uv run --with matplotlib python -m scripts.report_experiments \
  --run-root runs/a1-lr --output-dir results/a1-lr-plots --plot
```

Figures show recorded losses against gradient steps, active wall-clock seconds and tokens. Resume preserves active elapsed time from the checkpoint and excludes downtime between processes. When resuming an older checkpoint, superseded metrics are archived before logging the new continuation. Reports leave unexecuted jobs as `not_run` and never synthesize missing measurements.

For each question, retain the source/configuration, data fingerprints, curves, final metrics or failure status, generated samples where required, and an explanation of the observation. Learning-rate stability, batch-size effects and ablation findings must come from measured runs.

## Compute decisions

Prepare data before paying for idle GPU time where possible. First measure a short run on the chosen machine, then estimate the full suite from observed throughput and validation/checkpoint overhead. A subprocess timeout stops that process; it does not stop billing for an external instance.

The leaderboard uses a separate B200 hardware and 45-minute rule. An A40/4090 run is not an official B200 leaderboard result. Leaderboard submission and any new architecture modification remain separate decisions; nothing here provisions hardware, submits to the leaderboard or spends money.
