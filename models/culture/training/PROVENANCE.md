# Where the culture adapters came from

The LoRA adapters under `models/culture/<model>/<culture>/` were not trained in this
repository. They come from a separate one, and this directory records enough to say
exactly which run produced which adapter.

The training source itself is deliberately **not** copied here. A second copy of
`src/training/` that nothing in this repository executes would drift from the original
without anyone noticing, and the copy would still not be runnable — it needs that
repository's data, its MLflow store, and its own lockfile. What reproducibility needs is
the pin and the configuration, which is what follows.

## Upstream

| | |
|---|---|
| Repository | `git@github.com:neemiasbsilva/culture-mllm.git` |
| Commit | `c449dfc6434a1d06ee0ab55f1a0d2139f26ddc9e` (2026-06-03) |
| Training entrypoint | `src/training/train_hf.py` |
| Driver | `scripts/02_train_culture_models.sh` |
| Configs | `configs/gemma4_31b.yaml`, `configs/gemma4_e4b.yaml` — copied verbatim into `configs/` beside this file |
| Experiment tracking | MLflow, experiment `culture_mllm_training` |

Framework versions, as recorded in each adapter's own `README.md`:
PEFT 0.19.1, TRL 1.4.0, Transformers 5.8.1, PyTorch 2.12.0, Datasets 4.8.5,
Tokenizers 0.22.2.

`qwen3_vl_8b` is registered in `src/culture/registry.py` against the unquantized
`Qwen/Qwen3-VL-8B-Thinking` release, but its adapters are still being trained upstream.
Its config joins this directory, and its rows join the tables below, when a healthy
adapter is staged.

## What the runs held equal

Both models were trained on the same data for the same budget, so differences in
outcome are not differences in training effort:

| | gemma4_31b | gemma4_e4b |
|---|---|---|
| `max_epochs` | 250 | 250 |
| Stopped at | step 3900, epoch 32.7788 | step 3900, epoch 32.7788 (german 4400, 36.9820) |
| Effective batch | 16 (1 × 16) | 16 (2 × 8) |
| LoRA | r=8, alpha=16, q/k/v/o_proj | r=8, alpha=16, q/k/v/o_proj |
| `max_grad_norm` | 0.3 | 0.3 |
| `lr_schedule` | cosine | cosine |
| `dtype` | `bfloat16` | `bfloat16` |
| `quantization` | `4bit` (QLoRA) | none |
| `learning_rate` | 1.0e-4 | 2.0e-4 |

## Adapter health

`./run_experiment_add_culture.sh health` reads the trainer state and the weights of every
staged adapter, writes `outputs/culture/adapter_health.csv`, and
`figures/culture/summary/fig_adapter_health` charts it. An adapter is called healthy only
on evidence — a first logged loss below `ln(vocab_size)`, a bounded update norm, and a
held-out token accuracy far above guessing.

Do not read a newly staged adapter's numbers until `adapter_health.csv` says `healthy`
for it.

## MLflow run per adapter

Each staged adapter also carries its own `mlflow_run_id.txt`.

| culture | gemma4_31b | gemma4_e4b |
|---|---|---|
| german | `489588dcbff64bc39a5c138a11e18bd4` | `2129abea05274bc093ba5593f5abd5fd` |

Only the german adapters are staged; the other eight cultures' runs live upstream in the
same MLflow experiment and stage the same way when needed.

`models/culture/ADAPTERS.json` carries the SHA-256, LoRA configuration and health record
of every staged adapter, written by `./run_experiment_add_culture.sh adapters`.
