# Towards Socially Grounded AI Safety

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-locked-DE5FE9?logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.8-FFD21E?logo=huggingface&logoColor=black)
![Ruff](https://img.shields.io/badge/Ruff-passing-D7FF64?logo=ruff&logoColor=black)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2)
![tests](https://img.shields.io/badge/tests-390-4c1)

Reproduction of *Machine Bias: How Do Generative Language Models Answer Opinion
Polls?* (Boelaert, Coavoux, Ollion, Petev and Präg, *SMR* 2025), extended to
culture-finetuned LLMs.

This README is how to run the code. The method, the metrics and the findings are
in the paper.

## Install

```bash
uv sync --locked --group dev --extra culture --extra hub
```

`uv sync` is exact, so name every extra in one command: `culture` (experiments
3–4 and every notebook redraw), `hub` (the dataset), `inference` (experiment 2),
`api` (experiment 5), `muse` (`muse_glimmer_30b` alone).

The analysis reads a small part of the upstream replication package:

```bash
uv run --no-sync python -m machine_bias_reproduction hub pull --groups upstream
tar xzf hub/downloads/upstream/redraw-subset.tar.gz
```

## Experiments

| # | Experiment | Command | Needs |
| --- | --- | --- | --- |
| 1 | **Archived** — reanalyse the paper's model outputs | `./run_experiment.sh` | nothing |
| 2 | **Fresh** — regenerate Mixtral's answers | `./run_experiment.sh fresh --model …` | 26.4 GB GGUF |
| 3 | **Cultural** — culture-finetuned LLMs on each base | `./run_experiment_add_culture.sh sweep` | CUDA GPU |
| 4 | **Base models** — the same bases, un-finetuned | `./run_experiment_base_models.sh run --models all` | CUDA GPU |
| 5 | **Served** — proprietary models through the OpenAI API | `./run_experiment_add_culture.sh served-smoke` | API key |

```bash
./run_experiment.sh archived --questions all
./run_experiment.sh verify && ./run_experiment.sh test && ./run_experiment.sh lint

./scripts/download_model.sh
./run_experiment.sh fresh --model models/mixtral-8x7b-v0.1.Q4_K_M.gguf --mode all
./run_experiment.sh compare --question d_happy

./run_experiment_add_culture.sh adapters
./run_experiment_add_culture.sh sweep --models all --cultures german spanish-mx \
                                      --questions all --replicates 3
./run_experiment_add_culture.sh sweep --models qwen3_vl_2b --cultures global
./run_experiment_add_culture.sh sweep --models qwen3_vl_2b --cultures subpop
./run_experiment_add_culture.sh compare

./run_experiment_base_models.sh run --models all --replicates 3
./run_experiment_base_models.sh summary

./run_reports.sh
```

`sweep` drives one model, culture and question per invocation, skips completed
runs and logs one file each, so a failing run never takes the grid with it;
detach it with `screen`. `--replicates N` repeats every cell under its own seed
stream into `rep<k>/` directories, resampling full answers alone. `--cultures
global` is the distribution-matched arm, staged from
`checkpoints/global/<model>/distributional`, and `--cultures subpop` the
subgroup-matched arm, staged from `checkpoints/subpop/<model>/subpop`; both are
named explicitly.

Experiment 5 needs `OPENAI_API_KEY` and one model id per served model in `.env`
(see `.env.example`); naming a served model is what starts a billable run.

## Artifacts

Tables land in `outputs/`, plates in `figures/`, both under
`<source>/<question>/`. The notebooks in `notebooks/` redraw the population
plates and write the population tables; `./run_reports.sh` writes the paper's
table and figure set.

## Dataset

Published (private now) as
[MInDS-lab-UTFPR/TowardsSociallyGroundedAISafety](https://huggingface.co/datasets/MInDS-lab-UTFPR/TowardsSociallyGroundedAISafety):
every derived table under `outputs/`, the raw per-prompt records as parquet, and
the upstream subset. Public, so `pull` needs no token.

```bash
uv run --no-sync python -m machine_bias_reproduction hub status
uv run --no-sync python -m machine_bias_reproduction hub pull
rsync -a hub/downloads/data/ outputs/
```

The survey data are the World Values Survey's, and the replication package terms
govern their reuse.

## Citation

```
TODO
```

## Acknowledgments

```
TODO
```
