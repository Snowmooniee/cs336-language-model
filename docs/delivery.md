# Inference artifact and reproduction

## Artifact status

The real TinyStories batch-64 checkpoint has been exported and verified locally. The archive contains exactly six files: `weights.pt`, `config.json`, `vocab.pkl`, `merges.pkl`, `manifest.json` and `SHA256SUMS`. It excludes optimizer/RNG state and training text. Model weights are not included in the source repository, and no public weights download is available.

`weights.pt` is a `torch.save` dictionary containing only `model`, compatible with existing `load_model` and generation commands. All 22,696,448 parameters match the original checkpoint. It is **90,800,046 bytes**, versus 272,407,716 bytes for the training checkpoint. The ZIP is 91,049,851 bytes.

| File | SHA-256 |
| --- | --- |
| Original training checkpoint | `c41dbd96c25aa836dbd13c58fd3113fd6acbd3b2d09699e6e51878d699739789` |
| Inference-only weights | `9588b24832531c1f2c1fa4faab3490ca5676a7db8e548c1cfcd6aecc19b74c24` |
| Inference ZIP | `139945bb7c6caafdceec00c0780876e490c84804f4f8db2969ab8a5d810249b4` |

Different files have different hashes; parameter equality establishes correspondence. The manifest's `source_revision` is the supplied project compatibility baseline (`493654d`), not the exact historical training source. The original batch-64 run has source fingerprint `cd3621e4c26d88cbe6052a7fb5259cd5c2a97e32914aa62fa2ec05b1811fe104` in the experiment bundle. Export/boundary benchmark code is commit `794da624b87390e48453bab629c853ce257f9cd5`.

Those project revisions are provenance identifiers from the development repository. This source snapshot carries the implementation and evidence without that repository's commit history. The pinned Stanford revision is available at the [upstream repository](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712).

## Fresh-directory verification

Source was extracted from that commit's Git archive into a new directory; the artifact was copied into `model/`. Imports were checked to resolve inside the extracted directory. The pinned dependency environment was reused: this is **not** a fresh-machine dependency-installation claim.

The CPU run verified hashes, generated 64 identical greedy tokens with and without caching, continued the saved finished story to real EOS, and ran five workloads with three paired repeats. See [verification](../benchmarks/kv-cache/delivery-verification-20261026.json), [generation](../benchmarks/kv-cache/generation-cpu-20261026.json) and [raw boundary measurements](../benchmarks/kv-cache/cpu-boundaries-20261026.json).

## Reproduce a received artifact

The commands below require the matching model archive; cloning source alone does not provide trained weights. With the archive available, run from the project root on Linux/WSL:

```bash
uv sync --frozen
unzip tinystories-inference-20261026.zip -d artifacts/tinystories
(cd artifacts/tinystories && sha256sum -c SHA256SUMS)

uv run --frozen python -m scripts.generate \
  --checkpoint artifacts/tinystories/weights.pt \
  --vocab artifacts/tinystories/vocab.pkl \
  --merges artifacts/tinystories/merges.pkl \
  --prompt "Once upon a time" --max-new-tokens 64 --temperature 0 \
  --kv-cache --output results/received-model-generation.json

uv run --frozen python -m scripts.benchmark_inference \
  --checkpoint artifacts/tinystories/weights.pt \
  --vocab artifacts/tinystories/vocab.pkl \
  --merges artifacts/tinystories/merges.pkl \
  --cases-file benchmarks/kv-cache/cases-20261026.json \
  --allow-context-rebuilds --warmup 1 --repeats 3 --threads 4 \
  --device cpu --output results/received-model-boundaries.json
```

Choose new output paths. Remove `--kv-cache` with a different output path for uncached generation. `--allow-context-rebuilds` explicitly includes full-window reconstruction in timings.

## Export another checkpoint

```bash
uv run --frozen python -m scripts.export_inference \
  --checkpoint runs/a1-batch/tinystories-batch-64/checkpoint.pt \
  --vocab results/tinystories-train/vocab.pkl \
  --merges results/tinystories-train/merges.pkl \
  --source-revision YOUR_PROJECT_REVISION \
  --expected-checkpoint-sha256 YOUR_CHECKPOINT_SHA256 \
  --output-dir results/inference-export
```

The exporter checks compatibility/finite parameters, excludes training-only configuration, compares every exported tensor, writes new hashes and refuses an existing destination. The historical run/source fingerprint should accompany distribution.

## Access and checks

The [project report](showcase.html), training figures, CSV summaries and raw inference measurements are included with the source. Training reports can be rebuilt without a GPU, checkpoint or dataset download. Generation and inference benchmarking require a trained checkpoint or the matching exported model archive.

[CPU checks](../.github/workflows/cpu-checks.yml) is a manual workflow using the existing lock, pytest and Ruff. Local validation is **103 passed, 1 xpassed**, Ruff passed. The workflow has not been dispatched or verified remotely and does not start on push. No CI-green claim is made. Configuration follows [setup-uv](https://github.com/astral-sh/setup-uv) and [checkout](https://github.com/actions/checkout).
