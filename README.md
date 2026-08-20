# Towards Socially Grounded AI Safety

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-locked-DE5FE9?logo=uv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.3-150458?logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.17-8CAAE6?logo=scipy&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-3d5a80)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.11-11557C)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.8-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-0.19-FFD21E?logo=huggingface&logoColor=black)
![llama.cpp](https://img.shields.io/badge/llama.cpp-0.3.1-lightgrey)
![Ruff](https://img.shields.io/badge/Ruff-passing-D7FF64?logo=ruff&logoColor=black)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2)
![tests](https://img.shields.io/badge/tests-169%20passing-4c1)

Reproduction of *Machine Bias: How Do Generative Language Models Answer Opinion
Polls?* (Boelaert, Coavoux, Ollion, Petev and Präg, *SMR* 2025), extended to
culture-finetuned LLMs.

```text
.
├── run_experiment.sh                 experiments 1 and 2
├── run_experiment_add_culture.sh     experiment 3
├── run_experiment_base_models.sh     experiment 3's un-finetuned baseline
├── run_reports.sh                    the paper's tables and figures
│
├── src/
│   ├── machine_bias_reproduction/    core reproduction
│   │   ├── questions.py              the four outcomes and their encodings
│   │   ├── prompts.py                the paper's prompts, byte for byte
│   │   ├── data.py                   WVS loading, alignment, coverage
│   │   ├── metrics.py                nEMD, EMD, KL, JS, MMD
│   │   ├── analysis.py               distances, baselines, regressions, manifest
│   │   ├── figures.py                per-run plates
│   │   ├── inference.py              resumable traced generation, llama.cpp
│   │   ├── comparison.py             archived-vs-fresh significance testing
│   │   ├── verification.py           the paper's published checkpoints
│   │   └── provenance.py  io_utils.py  r_rng.py  config.py  cli.py
│   │
│   ├── culture/                      experiment 3
│   │   ├── registry.py               models, arms, run layout
│   │   ├── adapters.py               staging LoRA weights, ADAPTERS.json
│   │   ├── backend.py                batched transformers NTP and FA
│   │   ├── runner.py                 one model/culture/question run
│   │   ├── capacity.py               can this arm answer the paper's prompt?
│   │   ├── matching.py               which cultures the survey can speak to
│   │   ├── plate.py                  shared scaffolding for the culture plates
│   │   ├── *_panels.py               capacity, density, ranking, country, response
│   │   └── figures.py                cross-culture figures and reports
│   │
│   └── reports/                      the paper's published report set
│       ├── series.py  load.py  tables.py  report.py
│       ├── figures_main.py           Figures 2, 3, 4, 6
│       ├── figures_appendix.py       Figures S1, S2, S4, S9, S10, S15
│       ├── robustness.py             Tables S6, S7; Figures S12, S13, S14
│       ├── discriminator.py          Table S8
│       ├── bootstrap.py              Figure S16
│       └── backtranslation.py        Table S9; Figures S17, S18
│
├── scripts/                          shared shell helpers, the sweep driver, the model download
├── tests/                            unit, integration, golden
│
├── outputs/    <source>/<question>/ + reports/ (raw/, logs/, reports/tex/ ignored)
├── figures/    same layout from `outputs/` + reports/
├── models/     gitignored   weights
└── upstream/   gitignored   replication package
```

## Contents

- [Install](#install)
- **Experiments**
  - [1 — Archived](#1--archived)
  - [2 — Fresh](#2--fresh)
  - [3 — Cultural](#3--cultural)
    - [Prompt fidelity](#prompt-fidelity)
    - [Culture-matched subsets](#culture-matched-subsets)
  - [4 — Base models](#4--base-models)
- **Reference**
  - [Topics](#topics)
  - [Paper report set](#paper-report-set)
  - [Distances](#distances)
  - [Comparison](#comparison)
  - [Citation & Acknowledgments](#citation--acknowledgments)

Four experiments, run in this order:

| # | Experiment | Command | Needs |
| --- | --- | --- | --- |
| 1 | **Archived** — reanalyse the paper's model outputs | `./run_experiment.sh` | nothing |
| 2 | **Fresh** — regenerate Mixtral's answers | `./run_experiment.sh fresh --model …` | 26.4 GB GGUF |
| 3 | **Cultural** — german-finetuned LLMs on each base | `./run_experiment_add_culture.sh sweep` | CUDA GPU |
| 4 | **Base models** — the same bases, un-finetuned | `./run_experiment_base_models.sh run --models all` | CUDA GPU |


---

## Install

```bash
uv sync --locked --group dev
```

Also needs the upstream replication package extracted to
`upstream/extracted/Machine-Bias-replication/`. It is **not** versioned here
(736 MB archived, ~790,000 files); `upstream/MANIFEST.json` holds the SHA-256 of
every file consumed, so a separately obtained copy verifies against this repo.

---

## 1 — Archived

Reanalyse the outputs shipped with the paper. No model, no generation, seconds.
**Start here**: this is what proves the implementation matches the published
numbers.

```bash
./run_experiment.sh
```

| Checkpoint | Expected |
| --- | ---: |
| WVS observations | 26,981 |
| Unique NTP profiles | 13,904 |
| Happiness subpopulations | 687 |
| Mean valid NTP token mass | 0.984541778 |
| Overall NTP nEMD | 0.030770574 |
| Overall FA nEMD | 0.035423290 |
| Median pairwise WVS nEMD | 0.114638448 |
| Median pairwise NTP nEMD | 0.032728912 |

The paper publishes these for **happiness and Mixtral only**, so they are
asserted for that combination alone.

```bash
./run_experiment.sh archived --questions all   # all four topics
./run_experiment.sh verify [--full]            # --full CRCs the ZIP
./run_experiment.sh test
./run_experiment.sh lint
```

---

## 2 — Fresh

Regenerate Mixtral's answers instead of reading the archived ones.

```bash
./scripts/download_model.sh                    # download + SHA-256 verify
uv sync --locked --group dev --extra inference

./run_experiment.sh fresh --model models/mixtral-8x7b-v0.1.Q4_K_M.gguf --mode ntp --limit 2
./run_experiment.sh fresh --model models/mixtral-8x7b-v0.1.Q4_K_M.gguf --mode all
```

Resumable: every prompt writes its own atomic record. NTP uses the original
single-token, top-1,000-logprob method; FA uses temperature `0.7`, 12 tokens max,
strict validation, and a per-prompt derived seed so resuming cannot change
earlier answers (`--legacy-unseeded-fa` restores the paper's behaviour).

<details>
<summary><b>Inference looks stuck? Build llama-cpp-python with CUDA</b></summary>

`uv sync --extra inference` builds with **no GPU backend**. The driver's
`nvidia-smi` check only decides how many layers it *asks* llama.cpp to offload —
it cannot enable support that wasn't compiled in, so inference silently runs on
CPU.

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  uv sync --locked --group dev --extra inference \
  --reinstall-package llama-cpp-python --no-cache
```

`--reinstall-package … --no-cache` is required even on a repeat run: uv's build
cache is keyed on package/version, not `CMAKE_ARGS`.

If `nvcc --version` predates your GPU (e.g. RTX 50-series with a CUDA 12.4–12.6
toolkit, which doesn't know `sm_120`), target a virtual architecture instead:
add `-DCMAKE_CUDA_ARCHITECTURES=90-virtual` to `CMAKE_ARGS`.

Confirm CUDA actually linked before a full run:

```bash
strings .venv/lib/python3.11/site-packages/llama_cpp/lib/libllama.so | grep -i cublas
```

No output means it still isn't compiled in. Override with `--gpu-layers -1|0`.

</details>

<details>
<summary><b>Per-inference traceability</b></summary>

Each prompt writes one record under
`outputs/fresh/<question>/raw/<mode>/<hash>.json` carrying: the exact prompt and
its SHA-256; `run_id`, timings; the model path and SHA-256 actually loaded; the
backend build (version, compiled features, `gpu_offload_supported`,
`n_gpu_layers`, `n_ctx`, seed); the sampling parameters; `code_revision`; and for
FA **every** attempt — raw text, seed, parsed value, acceptance — not just the
accepted answer.

A resumed run reuses earlier records, so this per-record copy, not the run
manifest, identifies the build behind any individual answer. Two append-only
logs under `logs/` survive resumes: `inference_events.jsonl` and `runs.jsonl`.

```bash
./run_experiment.sh trace --question d_happy
```

Rebuilds every prompt, re-hashes against the stored record, writes
`inference_trace.csv`. Exits non-zero on any mismatch or `.failed.json`.

</details>

---

## 3 — Cultural

Culture-finetuned LLMs on each multimodal base — `german` on all four — sent
the **same prompts** as experiments 1 and 2. The other culture-MLLM
finetunings are out of scope for now: their finetuning datasets sample a
different population than the WVS respondents a distance is read against, so
the comparison would not be robust. `german` is the culture whose training
data and survey block draw on the same country's samples — Germany — so model
and survey describe the same people.

| Model key | Base model | Loading |
| --- | --- | --- |
| `gemma4_31b` | `google/gemma-4-31B-it` | 4-bit NF4 (QLoRA, as trained) |
| `gemma4_e4b` | `google/gemma-4-E4B-it` | bf16, unquantized (as trained) |
| `qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Thinking` | bf16, unquantized |
| `muse_glimmer_30b` | `meta-models/Muse-Glimmer-30B` | 4-bit NF4 (QLoRA, as trained) |

Every row loads the way it was finetuned, so no distance is read
through a forward pass its finetuning never saw. All four are original,
unquantized releases (gemma4_31b and muse_glimmer_30b are served NF4 because
that is how their QLoRA finetuning was fitted), so a gap between models is never
a quantization artifact.

`muse_glimmer_30b` is finetuned for `german` only, and its architecture needs a
newer Transformers than the other three trained against, so it runs alone: the
drivers select the `muse` extra (Transformers 5.15.0) for it and the `culture`
extra (5.8.1) for everything else automatically. `sweep --models all` still
works — each run is its own invocation.

```bash
uv sync --locked --group dev --extra culture

./run_experiment_add_culture.sh adapters                       # stage weights
./run_experiment_add_culture.sh smoke --models gemma4_31b --cultures german

# first comparison — 4 runs: the german culture on all four topics
./run_experiment_add_culture.sh sweep --models gemma4_31b --questions all

./run_experiment_add_culture.sh sweep --models all --questions all     # the rest
./run_experiment_add_culture.sh compare                        # figures + reports
```

**Start with that sweep**: `german` is the culture this repository's results argue
from, and it has a WVS respondent block, so its distance reads against people who
share the culture's language — see
[Culture-matched subsets](#culture-matched-subsets). The sweep defaults to
`german`.

`sweep` drives **one model/culture/question per invocation**, so one failing run
never takes the grid with it: one log per run, outcome and answered share in
`sweep_summary.tsv`, completed runs skipped (`--redo` to force), comparison rebuilt
at the end. Each run is 13,904 NTP + 26,981 FA prompts; interrupting is safe.

### Prompt fidelity

The culture-finetuned LLMs get the paper's prompt **verbatim** — the same `prompt_records`
builder experiments 1 and 2 use, not the chat contract they were fine-tuned on.
That is what makes a culture distance comparable with the paper's own number, and
it is why capacity is worth measuring.

Whether a model *can* answer it is measured every run, not assumed — capacity
varies by base model and question, and the runs that fail are exactly the ones
a distance would misrepresent. Each run classifies every prompt **valid** /
**invalid** / **failed** in `capacity.csv`, keeps the first 25 rejected generations
per mode in `capacity_examples.jsonl`, and charts answered share and NTP mass
against the 10% threshold in `fig_culture_capacity`. Distances use only
subpopulations with ≥20 valid answers, and `coverage` in `run_manifest.json`
records what was dropped. **A run that retains nothing still completes** — a model
that cannot answer is a result, not an error.

### Culture-matched subsets

`compare` emits a matched view under `figures/culture/<model>/<question>/matched/`
when the culture has a WVS respondent block: for `german`, Germany's respondents,
so the matched distance reads against people who share the culture's language. A
culture without a block is excluded from the matched view rather than shown
against a mismatched comparison. All-culture figures are kept as the superset,
and the country heatmap bolds only matched cells and never reorders columns to
imply a diagonal the data cannot support.

<details>
<summary><b>Finetuning internals and detached sweeps</b></summary>

LoRA was fitted on the text attention projections (`q/k/v/o_proj`, `r=8`,
`alpha=16`); `exclude_modules` in `adapter_config.json` names **which modules got
LoRA weights**, not a discarded vision stack. Inference loads the full multimodal
checkpoint via `AutoModelForImageTextToText` and composes the adapted text layers
on top; these prompts carry no image, so the vision tower is present but not
invoked.

`adapters` copies only the end-of-training finetuned weights, excluding per-step
`checkpoint-N` optimizer state (1.4 GB staged instead of 92 GB).
`ADAPTERS.json` records each SHA-256, declared base and LoRA config; a repeat run
re-hashes what is staged, so a truncated copy is replaced rather than trusted.

Each base loads once per invocation and every culture finetuning attaches to
it, so switching culture is a `set_adapter` call. Generation is batched
(`--batch-size`, default 4 for the NF4 pair gemma4_31b and muse_glimmer_30b,
16 for gemma4_e4b and qwen3_vl_8b, halved on CUDA OOM).

A sweep is 4 runs of ~41,000 prompts per model, so detach it:

```bash
screen -S culture
./run_experiment_add_culture.sh sweep --models gemma4_31b --questions all
# Ctrl-A then D to detach; screen -r culture to return

tail -f outputs/culture/logs/gemma4_31b-german-d_happy.log
ls outputs/culture/gemma4_31b/german/d_happy/raw/ntp | wc -l   # of 13,904
```

`gemma4_31b` occupies ~18 GB in NF4; nothing else should compete for the card.

</details>

---

## 4 — Base models

The same bases, un-finetuned, sent the same prompts. Every distance above is
otherwise read against **Mixtral**, which is a *different* base model, so the
comparison confounds the culture finetuning with the base model it was fitted on.
This arm is the same weights before finetuning — the only reference that isolates
what the finetuning did.

```bash
uv sync --locked --group dev --extra culture

./run_experiment_base_models.sh run --models all --dry-run   # the 16 runs

# every base, all four topics, Germany's prompts first inside each
screen -S base
./run_experiment_base_models.sh run --models all
# Ctrl-A then D to detach; screen -r base to return

tail -f outputs/culture/logs/gemma4_31b-base-d_happy.log
column -t -s $'\t' outputs/culture/logs/base_summary.tsv    # per-run outcomes
./run_experiment_base_models.sh summary                     # rebuild the tables
```

The `run` word is required — `./run_experiment_base_models.sh --models all` exits
2, and a bare call defaults to `gemma4_31b` alone.

`run` drives **one model/question per invocation** — 16 runs, all of one topic
before the next — so one failure never takes the grid with it: one log per run at
`outputs/culture/logs/<model>-base-<question>.log`, outcomes in
`base_summary.tsv`, comparison rebuilt at the end. A run is complete once it has
written `capacity.csv`, so a base model that could not answer is not retried
(`--redo` forces it). Each run is 13,904 NTP + 26,981 FA prompts; interrupting is
safe. The driver refuses to start while another culture run holds the card
(`--allow-concurrent` overrides).

Results land at `outputs/culture/<model>/base/<question>/`, beside the
culture-finetuned runs, so `base` becomes another arm in every comparison figure.
Two artifacts exist only once it has run:

| Artifact | What it carries |
| --- | --- |
| `summary/base_deltas.csv` | per country: each arm's mean nEMD minus the base's **on the same subpopulations**, with `nEMD_center` and the country coefficients paired the same way. Negative `delta_nEMD` = the finetuning moved the model closer |
| `fig_home_advantage_base` | the home-advantage difference-in-differences taken against the model's own base rather than against Mixtral |

Coverage differs between arms — a run drops any subpopulation with fewer than 20
valid answers — so every delta is averaged over the intersection of what both arms
scored, and the row records how many that was. The United States is the design's
omitted reference country, so its `beta_country_*` cells are null, not zero.

Germany's prompts go first by default (`--no-priority` disables,
`--first-countries` renames) because Germany is `german`'s matched country — see
[Culture-matched subsets](#culture-matched-subsets) — so base-vs-`german` is the
country-matched delta this arm exists to supply. That is ordering, not selection:
nothing lands until the question's run finishes, but a run stopped part-way
already holds the country the German comparison needs.

---

## Topics

| Variable | Topic | Answers |
| --- | --- | ---: |
| `d_happy` | Happiness | 4 |
| `d_polpos` | Politics | 10 (numerical) |
| `d_religiousp` | Religion | 7 |
| `d_trust` | Trust | 2 |

```bash
./run_experiment.sh archived --questions d_polpos
./run_experiment.sh fresh --model models/…gguf --question d_trust
./run_experiment_add_culture.sh sweep --questions all
```

Artifacts are namespaced by question, so runs never collide.

**Politics is not symmetric with the others.** Its prompt closes with
`Answer: ` — trailing space — where categorical questions close with `Answer:`.
It is asked 0–9 for NTP so the answer is one digit token, but 1–10 for FA; the
two align by position, never by value. NTP scores bare digits for politics and
space-prefixed letters for everything else. All four questions × both modes are
pinned byte-for-byte by `tests/unit/test_prompts.py`.

Politics also has 48 subpopulations where nobody answered. Their target is
undefined, so they are dropped and counted in `coverage`, not silently averaged.

## Paper report set

```bash
./run_reports.sh              # everything, all four questions
./run_reports.sh main         # Tables 2–6, Figures 2, 3, 4, 6
./run_reports.sh appendix     # Tables S1, S2, S4 + Figures S1, S2, S4, S9, S10, S15
./run_reports.sh robustness   # Tables S6–S9 + Figures S12–S14, S16–S18
./run_reports.sh smoke
```

Loads no model. Covers all four questions × all six archived series (NTP:
GPT-4T, Llama-3-70B, Mixtral-8x7B; FA: GPT-3, Llama-3-70B, Mixtral-8x7B). CSV in
`outputs/reports/`, `\input`-able booktabs in `outputs/reports/tex/`, plates in
`figures/reports/`. The CSV and the plates are versioned; the booktabs are not,
being a second encoding of the CSV beside them that this driver rewrites on
every run.

Upstream prints its tables to the R console and never writes them, so the table
files are new output. Two quirks are corrected and noted in
`outputs/reports/REPORT.md`: Figure S1 is written to disk (upstream only prints
it), and Figure S12 gets its correct name (upstream saves it as
`Figure-S10-prompting-strategy-MDS.png`). Figures 1, 5, S3, S11 and Table 1 have
no upstream generator and are listed as known gaps.

## Distances

Every row of `subpopulation_distances.csv` carries five measures.

| Measure | Stands for | Notes |
| --- | --- | --- |
| `nEMD` | normalized [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) | **the paper's metric**; all checkpoints, bands and regressions use it |
| `EMD` | [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) (Wasserstein-1) | unnormalized, unit-spacing version of nEMD; comparable within a question only |
| `KL` | [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) | WVS vs. model, in bits; ε = 1e-9 smoothing, since survey proportions contain exact zeros |
| `JS` | [Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) | base 2; symmetric, bounded [0, 1], no smoothing |
| `MMD` | [Maximum Mean Discrepancy](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions) | RBF kernel over positions, σ = 1; charges less for adjacent confusions |

ε and σ are recorded in every `run_manifest.json`. Quality bands (0.05, 0.10,
0.15, 0.30) apply to nEMD only — they are calibrated on that scale and would be
an invented threshold on a divergence.

## Comparison

Once experiments 1 and 2 have both run for a question:

```bash
./run_experiment.sh compare --question d_happy
```

Swaps archived/fresh labels for whole country–survey waves, so cells from the
same sample are not treated as independent; p-values Holm-adjusted across NTP and
FA. Because failing to reject a difference does not establish sameness, a
cluster-robust equivalence test also runs (default ±0.005 nEMD, one tenth of the
narrowest quality band; `--equivalence-margin` to change).

## Citation

If you use this reproduction, please cite the original paper:

```
TODO
```

## Acknowledgments

```
TODO
```
