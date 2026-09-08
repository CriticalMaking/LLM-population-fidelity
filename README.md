# Towards Socially Grounded AI Safety

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-locked-DE5FE9?logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.8-FFD21E?logo=huggingface&logoColor=black)
![Ruff](https://img.shields.io/badge/Ruff-passing-D7FF64?logo=ruff&logoColor=black)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2)
![tests](https://img.shields.io/badge/tests-282%20passing-4c1)

Reproduction of *Machine Bias: How Do Generative Language Models Answer Opinion
Polls?* (Boelaert, Coavoux, Ollion, Petev and Präg, *SMR* 2025), extended to
culture-finetuned LLMs.

The method, the metrics and the findings are the paper's. This README is how to
run the code and where the artifacts land.

## Experiments

| # | Experiment | Command | Needs |
| --- | --- | --- | --- |
| 1 | **Archived** — reanalyse the paper's model outputs | `./run_experiment.sh` | nothing |
| 2 | **Fresh** — regenerate Mixtral's answers | `./run_experiment.sh fresh --model …` | 26.4 GB GGUF |
| 3 | **Cultural** — culture-finetuned LLMs on each base | `./run_experiment_add_culture.sh sweep` | CUDA GPU |
| 4 | **Base models** — the same bases, un-finetuned | `./run_experiment_base_models.sh run --models all` | CUDA GPU |
| 5 | **Served** — three proprietary models through the OpenAI API | `./run_experiment_add_culture.sh served-smoke` | API key |

## Install

```bash
uv sync --locked --group dev
```

| Extra | For |
| --- | --- |
| `hub` | the Hugging Face dataset and the upstream subset |
| `inference` | experiment 2 (llama.cpp) |
| `culture` | experiments 3 and 4, and every notebook redraw |
| `muse` | `muse_glimmer_30b` alone (Transformers 5.15) |
| `api` | experiment 5 |

`uv sync` is exact: naming fewer extras than the environment holds uninstalls the
rest, so name them all in one command.

The analysis also reads a small part of the upstream replication package — the
WVS respondent table and answer levels, the subpopulation cells, the archived
model responses, the linear baselines and the original code. The dataset
repository carries exactly that part:

```bash
uv sync --locked --group dev --extra hub
uv run --no-sync python -m machine_bias_reproduction hub pull --groups upstream
tar xzf hub/downloads/upstream/redraw-subset.tar.gz
```

It lands under `upstream/extracted/` and is enough for every figure and table
here. The full 736 MB package is versioned nowhere; `upstream/MANIFEST.json`
holds the SHA-256 of every file consumed, so a separately obtained copy verifies
with `./run_experiment.sh verify`. The survey data are the World Values Survey's,
and the replication package terms govern their reuse.

## Layout

```text
.
├── run_experiment.sh / _add_culture.sh / _base_models.sh / run_reports.sh
├── src/
│   ├── machine_bias_reproduction/   questions, prompts, data, metrics, analysis,
│   │                                figures, inference, comparison, verification
│   ├── culture/                     experiments 3–5: registry, finetuned weights,
│   │                                backend, runner, capacity, matching, plates
│   └── reports/                     the paper's published tables and figures
├── scripts/  tests/  notebooks/     helpers, the sweep driver, population plates
├── outputs/  figures/               <source>/<question>/ + reports/
└── models/   upstream/              gitignored: weights, replication package
```

---

## 1 — Archived

No model, no generation, seconds. This is what proves the implementation matches
the published numbers.

```bash
./run_experiment.sh                             # happiness, both modes
./run_experiment.sh archived --questions all    # all four topics
./run_experiment.sh verify [--full]             # --full CRCs the ZIP
./run_experiment.sh test
./run_experiment.sh lint
```

Asserted for happiness and Mixtral, the only combination the paper publishes:

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

## 2 — Fresh

```bash
./scripts/download_model.sh                     # download + SHA-256 verify
uv sync --locked --group dev --extra inference

./run_experiment.sh fresh --model models/mixtral-8x7b-v0.1.Q4_K_M.gguf --mode all
./run_experiment.sh trace --question d_happy    # re-hash every stored prompt
```

Resumable: every prompt writes its own atomic record under
`outputs/fresh/<question>/raw/<mode>/`, carrying the prompt hash, the backend
build actually loaded, the sampling parameters and every FA attempt.

<details>
<summary><b>Inference on CPU? Build llama-cpp-python with CUDA</b></summary>

`uv sync --extra inference` builds with no GPU backend, so inference silently
runs on CPU.

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  uv sync --locked --group dev --extra inference \
  --reinstall-package llama-cpp-python --no-cache
```

`--reinstall-package … --no-cache` is required even on a repeat run: uv's build
cache is keyed on package and version, not on `CMAKE_ARGS`. If `nvcc --version`
predates your GPU, add `-DCMAKE_CUDA_ARCHITECTURES=90-virtual`. Confirm it
linked — no output means it did not:

```bash
strings .venv/lib/python3.11/site-packages/llama_cpp/lib/libllama.so | grep -i cublas
```

</details>

## 3 — Cultural

Culture-finetuned LLMs on each base below, sent the same prompts as experiments 1
and 2. Two cultures are in scope, `german` and `spanish-mx`, each paired with the
WVS block of the country its training data draws on.

| Model key | Base model | Loading |
| --- | --- | --- |
| `gemma4_31b` | `google/gemma-4-31B-it` | 4-bit NF4 (QLoRA, as trained) |
| `gemma4_e4b` | `google/gemma-4-E4B-it` | bf16, unquantized (as trained) |
| `qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Thinking` | bf16, unquantized |
| `qwen3_vl_2b` | `Qwen/Qwen3-VL-2B-Thinking` | bf16, unquantized |
| `llama3_2_3b` | `meta-llama/Llama-3.2-3B` | bf16, unquantized |
| `muse_glimmer_30b` | `meta-models/Muse-Glimmer-30B` | 4-bit NF4 (QLoRA, as trained) |

Every row loads the way it was finetuned, so no gap between models is a
quantization artifact.

```bash
uv sync --locked --group dev --extra culture

./run_experiment_add_culture.sh adapters                       # stage weights
./run_experiment_add_culture.sh smoke --models gemma4_31b --cultures german
./run_experiment_add_culture.sh sweep --models gemma4_31b --questions all
./run_experiment_add_culture.sh sweep --models all --questions all
./run_experiment_add_culture.sh compare                        # figures + reports
```

`sweep` drives one model, culture and question per invocation, so one failing run
never takes the grid with it: one log per run, outcomes in `sweep_summary.tsv`,
completed runs skipped (`--redo` forces). Each run is 13,904 NTP + 26,981 FA
prompts and interrupting is safe, so detach it with `screen`.

Models get the paper's prompt verbatim, not the chat contract they were finetuned
on, so capacity is measured rather than assumed: every prompt is classified
valid / invalid / failed in `capacity.csv`, rejected generations are kept in
`capacity_examples.jsonl`, and distances use only subpopulations with ≥20 valid
answers. A run that retains nothing still completes.

## 4 — Base models

The same bases, un-finetuned. Every other distance is read against Mixtral, a
different base, so this is the only reference that isolates what the finetuning
did.

```bash
./run_experiment_base_models.sh run --models all --dry-run   # the 16 runs
./run_experiment_base_models.sh run --models all
./run_experiment_base_models.sh summary                      # rebuild the tables

column -t -s $'\t' outputs/culture/logs/base_summary.tsv
```

The `run` word is required. Results land at
`outputs/culture/<model>/base/<question>/`, so `base` becomes another variant in
every comparison figure, and two artifacts exist only once it has run:

| Artifact | Carries |
| --- | --- |
| `summary/base_deltas.csv` | each variant's mean nEMD minus its base's, on the same subpopulations; negative `delta_nEMD` means the finetuning moved the model closer |
| `fig_home_advantage_base` | the home-advantage difference-in-differences against the model's own base |

## 5 — Served models

`luna`, `terra` and `sol` are three sizes of one proprietary reasoning family
reached through the OpenAI API. They run the base variant only and never join
`--models all`; naming one is what starts a billable run.

Put the key and one id per model in `.env` at the repository root
(`.env.example` is the template):

```
OPENAI_API_KEY=sk-...
LUNA_MODEL_ID=the-id-luna-is-served-under
TERRA_MODEL_ID=the-id-terra-is-served-under
SOL_MODEL_ID=the-id-sol-is-served-under
```

```bash
uv sync --locked --group dev --extra api

./run_experiment_add_culture.sh served-smoke          # all three, 5 prompts each
./scripts/sweep.sh --base --models luna --questions all
```

Read the smoke before paying for a run. A served chat model can answer correctly
and still score as a failure, so replies are split four ways:

| Verdict | Meaning |
| --- | --- |
| `answered` | the paper's exact format; a real run scores this |
| `wrapped` | a real answer the paper's parser rejects for its wrapping |
| `deflected` | the question handed back |
| `refused` | declines to answer from the profile |

`strict_rate` is what a run would score and `tolerant_rate` adds the wrapped
ones: a gap between them is a parser problem, a low `tolerant_rate` at every size
is a prompt-contract problem. Results land in
`outputs/culture/served_smoke/<question>/`, never inside a run directory.

---

## Topics

| Variable | Topic | Answers |
| --- | --- | ---: |
| `d_happy` | Happiness | 4 |
| `d_polpos` | Politics | 10 (numerical) |
| `d_religiousp` | Religion | 7 |
| `d_trust` | Trust | 2 |

Artifacts are namespaced by question, so runs never collide. Politics is not
symmetric with the others: it is asked 0–9 for NTP and 1–10 for FA, aligned by
position and never by value, and its 48 empty subpopulations are dropped into
`coverage` rather than averaged. All four questions in both modes are pinned
byte-for-byte by `tests/unit/test_prompts.py`.

## Distances

| Measure | Stands for | Notes |
| --- | --- | --- |
| `nEMD` | normalized [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) | **the paper's metric**; every checkpoint, band and regression uses it |
| `EMD` | [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) (Wasserstein-1) | unnormalized; comparable within a question only |
| `KL` | [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) | in bits; ε = 1e-9 smoothing |
| `JS` | [Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) | base 2; symmetric, bounded, no smoothing |
| `MMD` | [Maximum Mean Discrepancy](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions) | RBF kernel over positions, σ = 1 |

ε and σ are recorded in every `run_manifest.json`. Quality bands (0.05, 0.10,
0.15, 0.30) apply to nEMD alone.

## Population plates

Drawn straight from the run distributions, over every retained subpopulation of a
run — 687 cells where a run is complete, 639 on politics.

| Family | x | y |
| --- | --- | --- |
| `fig_population_adaptability_<view>_<mode>` | mean nEMD to the survey cells, `E` | adaptability ratio `A`, model dispersion over survey dispersion |
| `fig_population_structure_<view>_<mode>` | adaptability ratio `A` | survey-to-model correlation of pairwise distances, `rho` |
| `fig_population_center_<view>_<mode>` | distance to the pooled Germany cells | distance to every cell pooled |
| `fig_population_center_mexican_<view>_<mode>` | the same against Mexico | |

Every quantity is carried twice: the marker is the population value, and a tick
joined by a dotted rule is the same quantity on one reference country's cells
alone. The two center families draw no tick, their x axis already being the
reference. Each family is drawn in five views — `all`, plus one per topic — in
both modes, as `.png` and `.pdf`. Every dashed rule is a fixed reference, never a
fit.

| Reference | Country | Cells | Tick |
| --- | --- | ---: | --- |
| `_german` | Germany | 144 | bar |
| `_mexican` | Mexico | 131 | 45° slash |

| Variant | Fill |
| --- | --- |
| as released | hollow, and wears both ticks |
| `german` | solid, and wears Germany's tick |
| `spanish-mx` | bottom half filled (`///` hatch on bars), and wears Mexico's tick |

## Population fidelity

One score over the three readings the plates keep apart, then asked again inside
every group of subpopulations the survey distinguishes:

```
PFS = geometric_mean(accuracy, adaptability, structure)
    = geometric_mean(1 - E, min(A, 1/A), max(0, rho))
```

Cultural center alignment, `1 - C`, is reported beside PFS and never inside it;
`pfs_with_center` carries the four-term form for anyone who wants it folded in.
Seven group families are parsed out of the subpopulation label — country, survey
wave, sex, age band, education, employment, marital status — and every term is
recomputed inside each group rather than averaged down.

| Plate | Folder | Shows |
| --- | --- | --- |
| `fig_fidelity_ranking_<view>_<mode>` | `figures/fidelity/` | PFS per run, ranked |
| `fig_fidelity_components_<view>_<mode>` | `figures/fidelity/` | the three terms and the center behind each PFS |
| `fig_fidelity_center_<view>_<mode>` | `figures/fidelity/` | center alignment against PFS |
| `fig_fidelity_shift_<view>_<mode>` | `figures/fidelity/` | change in PFS against change in center |
| `fig_fidelity_cells_<view>_<mode>` | `figures/fidelity/` | the per-cell nEMD spread the accuracy term averages |
| `fig_fidelity_<family>_*_<view>_<mode>` | `figures/fidelity/<family>/` | that family's levels, components and shift |

Score heatmaps run 0–1 on `viridis`, so bright is good — the reverse of the error
heatmaps elsewhere. `max(0, rho)` is discontinuous at zero and the score is a
product, so a model correlating at `rho = -0.0001` scores exactly zero;
`binding_term` names the term that bound in every row.

## Grouped MDS plates

A sweep rebuilds each model's own MDS plates; the notebook adds the three
families that need more than one model in one picture, each a shared embedding on
one shared scale.

| Plate | Panels | Sample |
| --- | --- | --- |
| `fig_culture_mds_tier_{small,mid,big}` | one model per column, NTP and FA rows, base and `german` per panel | the Germany cells |
| `fig_culture_mds_base_families` | the untuned variant of every model, open first then served | the cells every drawn run shares |
| `fig_culture_mds_finetuned` | the six open models' `german` variants side by side | the Germany cells |

Size tiers are declared in `culture.registry.SIZE_TIERS`, so a new base picks up
a column by being added there.

## Notebooks

| Notebook | Writes |
| --- | --- |
| `population_adaptability.ipynb` | `population_adaptability.csv`, 60 files in `figures/` |
| `population_structure.ipynb` | `population_structure.csv`, 40 files in `figures/` |
| `population_fidelity_evaluation.ipynb` | `population_fidelity_{cells,groups,overall}.csv`, 520 files in `figures/fidelity/` |
| `mds_model_comparison.ipynb` | `culture_mds_groups.csv` and the grouped MDS plates |

Adaptability runs before structure, which joins its table; fidelity and MDS read
neither and can run alone.

```bash
uv run --no-sync --with nbclient --with ipykernel python - <<'PY'
import nbformat
from nbclient import NotebookClient

for path, timeout in (
    ("notebooks/population_adaptability.ipynb", 3600),
    ("notebooks/population_structure.ipynb", 3600),
    ("notebooks/population_fidelity_evaluation.ipynb", 7200),
    ("notebooks/mds_model_comparison.ipynb", 14400),
):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3").execute()
    nbformat.write(nb, path)
PY
```

`--no-sync` matters: a bare `uv run` re-syncs the environment and destroys the
hand-built CUDA `llama-cpp-python` wheel.

## Paper report set

```bash
./run_reports.sh              # everything, all four questions
./run_reports.sh main         # Tables 2–6, Figures 2, 3, 4, 6
./run_reports.sh appendix     # Tables S1, S2, S4 + Figures S1, S2, S4, S9, S10, S15
./run_reports.sh robustness   # Tables S6–S9 + Figures S12–S14, S16–S18
./run_reports.sh smoke
```

Loads no model. All four questions × all six archived series (NTP: GPT-4T,
Llama-3-70B, Mixtral-8x7B; FA: GPT-3, Llama-3-70B, Mixtral-8x7B). CSV in
`outputs/reports/`, plates in `figures/reports/`. `outputs/reports/REPORT.md`
notes the two upstream quirks corrected here and the artifacts with no upstream
generator.

## Comparison

Once experiments 1 and 2 have both run for a question:

```bash
./run_experiment.sh compare --question d_happy
```

Labels are swapped for whole country–survey waves, so cells from one sample are
never treated as independent, and p-values are Holm-adjusted across both modes. A
cluster-robust equivalence test runs beside it (default ±0.005 nEMD,
`--equivalence-margin` to change), because failing to reject a difference does
not establish sameness.

## Hugging Face dataset

Published as
[MInDS-lab-UTFPR/TowardsSociallyGroundedAISafety](https://huggingface.co/datasets/MInDS-lab-UTFPR/TowardsSociallyGroundedAISafety):
every derived table under `outputs/`, the raw per-prompt records as parquet, and
the upstream subset. Public, so `pull` needs no token.

```bash
uv sync --locked --group dev --extra hub
uv run --no-sync python -m machine_bias_reproduction hub status
uv run --no-sync python -m machine_bias_reproduction hub pull                # everything
uv run --no-sync python -m machine_bias_reproduction hub pull --groups raw   # the raw records
rsync -a hub/downloads/data/ outputs/                                        # beside each run's CSVs
```

Publishing needs a write-scoped token, entered once at the prompt and stored in
`~/.cache/huggingface/token` — never in a file or on a command line.

```bash
uv run --no-sync hf auth login
uv run --no-sync python -m machine_bias_reproduction hub push --upstream --dry-run
uv run --no-sync python -m machine_bias_reproduction hub push --upstream
```

`push` skips every chunk whose contents did not change, and publishes every run
present in `outputs/`, including one a sweep is still writing. `--upstream` packs
this checkout's upstream subset as well; publish it deliberately, since it
carries World Values Survey data.

---

## Citation

If you use this reproduction, please cite the original paper:

```
TODO
```

## Acknowledgments

```
TODO
```
