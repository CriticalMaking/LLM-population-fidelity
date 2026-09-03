# Towards Socially Grounded AI Safety

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-locked-DE5FE9?logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.8-FFD21E?logo=huggingface&logoColor=black)
![Ruff](https://img.shields.io/badge/Ruff-passing-D7FF64?logo=ruff&logoColor=black)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2)
![tests](https://img.shields.io/badge/tests-169%20passing-4c1)

Reproduction of *Machine Bias: How Do Generative Language Models Answer Opinion
Polls?* (Boelaert, Coavoux, Ollion, Petev and Präg, *SMR* 2025), extended to
culture-finetuned LLMs.

| # | Experiment | Command | Needs |
| --- | --- | --- | --- |
| 1 | **Archived** — reanalyse the paper's model outputs | `./run_experiment.sh` | nothing |
| 2 | **Fresh** — regenerate Mixtral's answers | `./run_experiment.sh fresh --model …` | 26.4 GB GGUF |
| 3 | **Cultural** — german-finetuned LLMs on each base | `./run_experiment_add_culture.sh sweep` | CUDA GPU |
| 4 | **Base models** — the same bases, un-finetuned | `./run_experiment_base_models.sh run --models all` | CUDA GPU |
| 5 | **Served** — three proprietary models through the OpenAI API | `./run_experiment_add_culture.sh served-smoke` | API key |

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

## Install

```bash
uv sync --locked --group dev
```

Also needs the upstream replication package. The analysis reads a small part of
it — the WVS respondent table and answer levels, the subpopulation cells, the
archived model response tables, the linear baselines and the original code —
and the dataset repository carries exactly that part as
`upstream/redraw-subset.tar.gz`. Pull it and extract it at the repository root:

```bash
uv sync --locked --group dev --extra hub
uv run --no-sync python -m machine_bias_reproduction hub pull --groups upstream
tar xzf hub/downloads/upstream/redraw-subset.tar.gz
```

It lands under the gitignored `upstream/extracted/Machine-Bias-replication/`
and is enough for every figure and table in this README. The full package is
**not** versioned anywhere (736 MB zip, ~790,000 files): `upstream/MANIFEST.json`
holds the SHA-256 of every file consumed, so a separately obtained copy verifies
against this repo with `./run_experiment.sh verify`, and the tests that read the
prompt trees and the zip skip until it is present. The survey data are the World
Values Survey's; the replication package terms govern their reuse.

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

`uv sync --extra inference` builds with **no GPU backend**, so inference silently
runs on CPU — the driver's `nvidia-smi` check only decides how many layers it
*asks* llama.cpp to offload.

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  uv sync --locked --group dev --extra inference \
  --reinstall-package llama-cpp-python --no-cache
```

`--reinstall-package … --no-cache` is required even on a repeat run: uv's build
cache is keyed on package/version, not `CMAKE_ARGS`. If `nvcc --version` predates
your GPU (e.g. RTX 50-series with a CUDA 12.4–12.6 toolkit, which doesn't know
`sm_120`), add `-DCMAKE_CUDA_ARCHITECTURES=90-virtual`. Confirm it linked — no
output means it didn't:

```bash
strings .venv/lib/python3.11/site-packages/llama_cpp/lib/libllama.so | grep -i cublas
```

</details>

<details>
<summary><b>Per-inference traceability</b></summary>

Each prompt writes one record under
`outputs/fresh/<question>/raw/<mode>/<hash>.json`: the prompt and its SHA-256,
timings, the model and backend build actually loaded, the sampling parameters,
`code_revision`, and for FA **every** attempt. A resumed run reuses earlier
records, so this per-record copy, not the run manifest, identifies the build
behind any individual answer.

```bash
./run_experiment.sh trace --question d_happy
```

Rebuilds every prompt, re-hashes against the stored record, writes
`inference_trace.csv`. Exits non-zero on any mismatch or `.failed.json`.

</details>

---

## 3 — Cultural

Culture-finetuned LLMs on each base below — `german` on all six — sent the
**same prompts** as experiments 1 and 2. Only `german` is in scope for now: its
training data and its WVS block draw on the same country's samples, so model and
survey describe the same people.

| Model key | Base model | Loading |
| --- | --- | --- |
| `gemma4_31b` | `google/gemma-4-31B-it` | 4-bit NF4 (QLoRA, as trained) |
| `gemma4_e4b` | `google/gemma-4-E4B-it` | bf16, unquantized (as trained) |
| `qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Thinking` | bf16, unquantized |
| `qwen3_vl_2b` | `Qwen/Qwen3-VL-2B-Thinking` | bf16, unquantized |
| `llama3_2_3b` | `meta-llama/Llama-3.2-3B` | bf16, unquantized |
| `muse_glimmer_30b` | `meta-models/Muse-Glimmer-30B` | 4-bit NF4 (QLoRA, as trained) |

Every row loads the way it was finetuned from an original release, so no distance
is read through a forward pass its finetuning never saw and no gap between models
is a quantization artifact. `qwen3_vl_2b` is `qwen3_vl_8b`'s architecture at a
quarter the size, separating finetuning from capacity; `llama3_2_3b` is the one
text-only base. `muse_glimmer_30b` is `german`-only and needs the `muse` extra
(Transformers 5.15.0) where the rest use `culture` (5.8.1); the drivers pick
either automatically, so `sweep --models all` still works.

```bash
uv sync --locked --group dev --extra culture

./run_experiment_add_culture.sh adapters                       # stage weights
./run_experiment_add_culture.sh smoke --models gemma4_31b --cultures german

# start here — 4 runs: the german culture on all four topics
./run_experiment_add_culture.sh sweep --models gemma4_31b --questions all

./run_experiment_add_culture.sh sweep --models all --questions all
./run_experiment_add_culture.sh compare                        # figures + reports
```

`sweep` drives **one model/culture/question per invocation**, so one failing run
never takes the grid with it: one log per run, outcomes in `sweep_summary.tsv`,
completed runs skipped (`--redo` to force), comparison rebuilt at the end. Each
run is 13,904 NTP + 26,981 FA prompts; interrupting is safe.

**Prompt fidelity.** These models get the paper's prompt **verbatim**, not the
chat contract they were finetuned on — which is what makes a culture distance
comparable with the paper's own number, and why capacity is measured rather than
assumed. Each run classifies every prompt **valid** / **invalid** / **failed** in
`capacity.csv`, keeps rejected generations in `capacity_examples.jsonl`, and
charts answered share and NTP mass against the 10% threshold. Distances use only
subpopulations with ≥20 valid answers; `coverage` records what was dropped. A run
that retains nothing still completes — a model that cannot answer is a result, not
an error.

**Culture-matched subsets.** `compare` emits a matched view under
`figures/culture/<model>/<question>/matched/` when the culture has a WVS
respondent block — for `german`, Germany's respondents. A culture without a block
is excluded rather than compared against a mismatched one, and the all-culture
figures stay as the superset.

<details>
<summary><b>Finetuning internals and detached sweeps</b></summary>

LoRA was fitted on the text attention projections (`q/k/v/o_proj`, `r=8`,
`alpha=16`); `exclude_modules` in `adapter_config.json` names **which modules got
LoRA weights**, not a discarded vision stack. Inference loads the full multimodal
checkpoint and composes the adapted text layers on top, and these prompts carry no
image, so the vision tower is never invoked.

`adapters` stages only the end-of-training weights (1.4 GB instead of 92 GB) and
`ADAPTERS.json` records each SHA-256, so a truncated copy is replaced rather than
trusted. Each base loads once per invocation and every culture finetuning attaches
to it, so switching culture is a `set_adapter` call. Generation is batched
(`--batch-size`, default 4 for the NF4 pair, 16 for `gemma4_e4b` and
`qwen3_vl_8b`, 32 for the small pair, halved on CUDA OOM).

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
otherwise read against **Mixtral**, a different base model, so the comparison
confounds the culture finetuning with the base it was fitted on. This is the only
reference that isolates what the finetuning did.

```bash
uv sync --locked --group dev --extra culture

./run_experiment_base_models.sh run --models all --dry-run   # the 16 runs

screen -S base
./run_experiment_base_models.sh run --models all
# Ctrl-A then D to detach; screen -r base to return

tail -f outputs/culture/logs/gemma4_31b-base-d_happy.log
column -t -s $'\t' outputs/culture/logs/base_summary.tsv    # per-run outcomes
./run_experiment_base_models.sh summary                     # rebuild the tables
```

The `run` word is required — without it the driver exits 2, and a bare call
defaults to `gemma4_31b` alone. It drives **one model/question per invocation** —
16 runs, all of one topic before the next — and a run is complete once it has
written `capacity.csv`, so a base that could not answer is not retried (`--redo`
forces it). It refuses to start while another culture run holds the card
(`--allow-concurrent` overrides), and Germany's prompts go first by default
(`--no-priority`, `--first-countries`) so a run stopped part-way already holds the
country the base-vs-`german` delta needs.

Results land at `outputs/culture/<model>/base/<question>/`, so `base` becomes
another variant in every comparison figure. Two artifacts exist only once it has
run:

| Artifact | What it carries |
| --- | --- |
| `summary/base_deltas.csv` | per country: each variant's mean nEMD minus the base's **on the same subpopulations**, `nEMD_center` and the country coefficients paired the same way. Negative `delta_nEMD` = the finetuning moved the model closer |
| `fig_home_advantage_base` | the home-advantage difference-in-differences against the model's own base rather than against Mixtral |

Coverage differs between variants, so every delta is averaged over the
intersection of what both scored and the row records how many that was. The
United States is the omitted reference country, so its `beta_country_*` cells are
null, not zero.

---

## 5 — Served models

`luna`, `terra` and `sol` are three sizes of one proprietary reasoning family
reached through the OpenAI API, and they answer one question: what a frontier
served model does on the prompts the open bases answer. There is no culture-MLLM
finetuning for them, so they run the **base variant only**, and they never join
`--models all` — naming one explicitly is what starts a billable run.

Put the key and one id per model in `.env` at the repository root (git-ignored;
`.env.example` is the template). Each model reads its own `<KEY>_MODEL_ID`, and a
missing one fails immediately with the variable named:

```
OPENAI_API_KEY=sk-...

LUNA_MODEL_ID=the-id-luna-is-served-under
TERRA_MODEL_ID=the-id-terra-is-served-under
SOL_MODEL_ID=the-id-sol-is-served-under
```

```bash
uv sync --locked --group dev --extra api

./run_experiment_add_culture.sh served-smoke          # all three, 5 prompts each
./run_experiment_add_culture.sh served-smoke --prompts 50 --questions all
./run_experiment_add_culture.sh served-smoke --reasoning-effort low

./scripts/sweep.sh --base --models luna --questions all --dry-run
./scripts/sweep.sh --base --models luna --questions all
```

### Read `served-smoke` before paying for a run

A served model is a chat model answering a prompt written for a completion model:
the parser accepts only the bare option text (`B. Quite happy`) on the first line,
so a chat model can answer correctly and still score as a failure — a different
problem from declining the persona, and with a different fix. The smoke splits
replies four ways:

| verdict | what it means |
| --- | --- |
| `answered` | the paper's exact format; a real run scores this |
| `wrapped` | a real answer the paper's parser rejects for its wrapping |
| `deflected` | the question handed back — "please choose one: A…D" |
| `refused` | declines to answer from the profile — "cannot be determined" |

`strict_rate` is what a run would score; `tolerant_rate` adds the wrapped ones. A
gap between the two is a parser problem; a low `tolerant_rate` at every size is a
prompt-contract problem, not a size problem. Each row also carries mean NTP answer
mass, whether the endpoint returned logprobs at all, and the reasoning effort with
the tokens it spent, so a difference between two variants is not read as size when
it is effort. Everything lands in `outputs/culture/served_smoke/<question>/` —
never inside a run directory, so it cannot make a sweep think a cell is finished.

<details>
<summary><b>Reasoning, NTP and the four honesty notes</b></summary>

`--reasoning-effort` (default `medium`) is sent on **every** request and held
equal across whichever models are compared — left unset, a difference between
variants could be effort rather than model. `--fa-max-tokens` is raised to 300
automatically, since `max_completion_tokens` counts reasoning tokens and the
paper's default of 12 would be spent before any visible answer.

NTP is computable in principle — the endpoint returns up to 20 next-token
candidates against at most 10 answers, so each letter's probability is read and
renormalized as the local backends do. Whether the mass is usable is decided by
the preflight probe that gates every run (32 prompts, 0.10 floor); if it comes
back empty the run drops NTP, records that under `modes_run`, and proceeds on full
answers alone, still comparable to the open bases on every FA row. The probe asks
for a **single** token, so at any effort above `none` the budget is spent
reasoning and no logprob comes back — the wiring, not a finding about the model,
and `logprobs_supported` tells the two apart.

Four honesty notes, all recorded in the manifest: the API's `seed` is best-effort,
so reproducibility is weaker than the local backends' (`seed_semantics`); the
prompt arrives as a single user message rather than a raw completion
(`prompt_contract`); and a model that refuses the sampling temperature or the
reasoning effort is called without it (`fa_temperature`, `reasoning_effort` beside
`reasoning_effort_requested`).

The `api` extra carries no PyTorch, so served models run apart from local ones —
the drivers refuse to mix them and select the extra automatically. Since syncing
swaps the environment, rebuild the CUDA `llama-cpp-python` wheel before the next
`fresh` run, and re-sync `--extra culture` or `--extra muse` before the next local
model.

</details>

---

## Topics

| Variable | Topic | Answers |
| --- | --- | ---: |
| `d_happy` | Happiness | 4 |
| `d_polpos` | Politics | 10 (numerical) |
| `d_religiousp` | Religion | 7 |
| `d_trust` | Trust | 2 |

Artifacts are namespaced by question, so runs never collide.

**Politics is not symmetric with the others.** Its prompt closes with `Answer: ` —
trailing space — where categorical questions close with `Answer:`. It is asked 0–9
for NTP so the answer is one digit token, but 1–10 for FA; the two align by
position, never by value. NTP scores bare digits for politics and space-prefixed
letters for everything else, and all four questions × both modes are pinned
byte-for-byte by `tests/unit/test_prompts.py`. Politics also has 48
subpopulations where nobody answered: their target is undefined, so they are
dropped and counted in `coverage`, not silently averaged.

## Population plates

Three plate families that two notebooks draw straight from the run distributions
rather than from the derived CSVs. They score **every retained subpopulation of a
run** — all five countries, 687 cells where a run is complete, 639 on politics —
and compare each base model against its own untuned variant.

| Family | x | y |
| --- | --- | --- |
| `fig_population_adaptability_<view>_<mode>` | mean nEMD to the survey cells | model dispersion over survey dispersion |
| `fig_population_center_<view>_<mode>` | distance to the pooled German cells | distance to every cell pooled |
| `fig_population_structure_<view>_<mode>` | model dispersion over survey dispersion | survey-to-model correlation of pairwise distances |

Every quantity is carried twice: the marker is the population value, and a tick
joined to it by a dotted rule is the same quantity on the 144 German cells alone.
An untuned variant's tick against its finetuned variant's tick is what the
culture-MLLM finetuning bought on the cells it was fitted for — and a marker that
moved one way while its tick moved the other bought Germany at the rest of the
world's expense. The centers family draws no tick, its x axis already being the
German reference; both CSVs keep both scopes, the German one in the `_german`
columns. Every dashed rule is a **fixed reference, never a fit**, so none of them
ever tilts.

Each family is drawn in five views: `all` (one panel per topic) and one per topic.
Distances scale with a topic's answer count, so panels share a y axis, never an x
scale.

```bash
uv run --no-sync --with nbclient --with ipykernel python - <<'PY'
import nbformat
from nbclient import NotebookClient

for path in ("notebooks/population_adaptability.ipynb",
             "notebooks/population_structure.ipynb"):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=3600, kernel_name="python3").execute()
    nbformat.write(nb, path)
PY
```

`--no-sync` matters: a bare `uv run` re-syncs the environment and destroys the
hand-built CUDA `llama-cpp-python` wheel. The adaptability notebook must run
first — the structure notebook joins its table. Both write to
`outputs/culture/population_*.csv` and the top level of `figures/`.

## Population fidelity

One score over the three population readings the plates above keep apart, from
the same run distributions, and then asked again inside every group of
subpopulations the survey distinguishes.

`PFS = geometric_mean(accuracy, dispersion, structure)`, each term pulled to
0-1 with higher better: `accuracy = 1 - E`, `dispersion = min(AR, 1/AR)`,
`structure = max(0, rho)`. Cultural center alignment, `1 - C`, is reported
beside it and never inside it — a model can relocate its average opinion toward
the target culture while staying flat across cells, and keeping the two apart is
what makes that visible. `pfs_with_center` carries the four-term form for anyone
who wants it folded in.

Seven group families are parsed out of the subpopulation label itself — country,
survey wave, sex, age band, education, employment, marital status — and every
term is recomputed **inside** each group, never averaged down from the pooled
figure. The upstream cell assignment is a matching rather than a strict
partition, so the label defines the group, not the respondents inside the cell.

| Plate | Folder | x / columns |
| --- | --- | --- |
| `fig_fidelity_ranking_<view>_<mode>` | `figures/fidelity/` | PFS per run, ranked |
| `fig_fidelity_components_<view>_<mode>` | `figures/fidelity/` | the three terms and the centre behind each PFS |
| `fig_fidelity_center_<view>_<mode>` | `figures/fidelity/` | centre alignment against PFS |
| `fig_fidelity_shift_<view>_<mode>` | `figures/fidelity/` | change in PFS against change in centre |
| `fig_fidelity_cells_<view>_<mode>` | `figures/fidelity/` | the per-cell nEMD spread the accuracy term averages |
| `fig_fidelity_<family>_<view>_<mode>` | `figures/fidelity/<family>/` | that family's levels |
| `fig_fidelity_<family>_components_<view>_<mode>` | `figures/fidelity/<family>/` | the same four terms, one panel each |
| `fig_fidelity_<family>_shift_<view>_<mode>` | `figures/fidelity/<family>/` | change in PFS against the untuned base |

Score heatmaps run 0-1 on `viridis`, so bright is good — the reverse of the
error heatmaps elsewhere, which run `viridis_r`. Five views and two modes as
above, `.png` and `.pdf` throughout.

The notebook asserts its own recomputation against the split per-run outputs
that `combined-metric-inputs.csv` was aggregated from: every per-cell `nEMD`
matches the run's `subpopulation_distances.csv` row for row, and the pooled row
reproduces `population_adaptability.csv`. One column deliberately differs —
`center_nEMD` there pools respondents, `c_center_nemd` here pools retained
cells.

`max(0, rho)` is discontinuous at zero and the score is a product, so a model
correlating at `rho = -0.0001` scores exactly zero. Nineteen of the 116 pooled
rows land there, every one on a `rho` between `-0.026` and `-0.0001`. Nothing
smooths it: `binding_term` names the term that bound in every row of every
table, and the ranking plate prints it beside each zero bar.

```bash
uv run --no-sync --with nbclient --with ipykernel python - <<'NB'
import nbformat
from nbclient import NotebookClient

path = "notebooks/population_fidelity_evaluation.ipynb"
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=7200, kernel_name="python3").execute()
nbformat.write(nb, path)
NB
```

Reads nothing the adaptability and structure notebooks write, so it can run on
its own. Writes `outputs/culture/population_fidelity_{cells,groups,overall}.csv`
and 520 files under `figures/fidelity/`.

## Grouped MDS plates

A sweep rebuilds each model's own MDS plates at the end of its run.
`notebooks/mds_model_comparison.ipynb` does that too, and then draws the three
families that need more than one model in the same picture — each one shared
embedding on one shared scale, so every panel of a plate is comparable.

| Plate | Panels | Sample |
| --- | --- | --- |
| `fig_culture_mds_tier_{small,mid,big}` | one model per column, NTP and FA rows, both variants per panel | the German cells |
| `fig_culture_mds_base_families` | the untuned variant of every model, open first then served | the cells every drawn run shares, all five countries |
| `fig_culture_mds_finetuned` | the six `german` variants side by side | the German cells |

Size tiers are declared in `culture.registry.SIZE_TIERS`, so a new base model
picks up a column by being added there. The served models have no published size
and are in no tier — the base-families plate is the only geometry they appear in,
and it is full answers throughout because those endpoints return no next-token
probabilities. A run that never happened contributes no panel and no row. Every
panel's mean nEMD lands in `outputs/culture/culture_mds_groups.csv`.

```bash
uv run --no-sync --with nbclient --with ipykernel python - <<'PY'
import nbformat
from nbclient import NotebookClient

path = "notebooks/mds_model_comparison.ipynb"
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=14400, kernel_name="python3").execute()
nbformat.write(nb, path)
PY
```

A full four-question pass is tens of minutes; every driver in the notebook takes
a question list, so `tier_plates(["d_happy"])` is the quick look.

## Paper report set

```bash
./run_reports.sh              # everything, all four questions
./run_reports.sh main         # Tables 2–6, Figures 2, 3, 4, 6
./run_reports.sh appendix     # Tables S1, S2, S4 + Figures S1, S2, S4, S9, S10, S15
./run_reports.sh robustness   # Tables S6–S9 + Figures S12–S14, S16–S18
./run_reports.sh smoke
```

Loads no model. Covers all four questions × all six archived series (NTP: GPT-4T,
Llama-3-70B, Mixtral-8x7B; FA: GPT-3, Llama-3-70B, Mixtral-8x7B). CSV in
`outputs/reports/`, `\input`-able booktabs in `outputs/reports/tex/`, plates in
`figures/reports/`; the booktabs alone are unversioned, being a second encoding of
the CSV that this driver rewrites on every run.

Upstream prints its tables to the R console and never writes them, so the table
files are new output. `outputs/reports/REPORT.md` notes the two upstream quirks
corrected here, and the five figures and one table with no upstream generator.

## Distances

Every row of `subpopulation_distances.csv` carries five measures.

| Measure | Stands for | Notes |
| --- | --- | --- |
| `nEMD` | normalized [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) | **the paper's metric**; all checkpoints, bands and regressions use it |
| `EMD` | [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) (Wasserstein-1) | unnormalized, unit-spacing; comparable within a question only |
| `KL` | [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) | WVS vs. model, in bits; ε = 1e-9 smoothing, since survey proportions contain exact zeros |
| `JS` | [Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) | base 2; symmetric, bounded [0, 1], no smoothing |
| `MMD` | [Maximum Mean Discrepancy](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions) | RBF kernel over positions, σ = 1; charges less for adjacent confusions |

ε and σ are recorded in every `run_manifest.json`. Quality bands (0.05, 0.10,
0.15, 0.30) apply to nEMD only — they are calibrated on that scale and would be an
invented threshold on a divergence.

## Comparison

Once experiments 1 and 2 have both run for a question:

```bash
./run_experiment.sh compare --question d_happy
```

Swaps archived/fresh labels for whole country–survey waves, so cells from the same
sample are not treated as independent; p-values Holm-adjusted across NTP and FA.
Because failing to reject a difference does not establish sameness, a
cluster-robust equivalence test also runs (default ±0.005 nEMD, one tenth of the
narrowest quality band; `--equivalence-margin` to change).

## Hugging Face dataset

The outputs are published as
[MInDS-lab-UTFPR/TowardsSociallyGroundedAISafety](https://huggingface.co/datasets/MInDS-lab-UTFPR/TowardsSociallyGroundedAISafety):
every derived table under `outputs/` plus the raw per-prompt records as parquet,
and, since the redraw needs it, the upstream subset described under Install.
All of it comes back with one command; `--groups` narrows it.

```bash
uv sync --locked --group dev --extra hub
uv run --no-sync python -m machine_bias_reproduction hub status
uv run --no-sync python -m machine_bias_reproduction hub pull                      # tables, raw parquet, card, upstream subset
uv run --no-sync python -m machine_bias_reproduction hub pull --groups upstream    # the upstream subset alone
tar xzf hub/downloads/upstream/redraw-subset.tar.gz
```

Publishing needs a write-scoped token (`hf auth login` or `HF_TOKEN`). `hub build`
mirrors `outputs/` into `hub/datasets/`; `hub push` builds and uploads it, skipping
every chunk whose contents did not change. `--upstream` on either packs the
upstream subset from this checkout's `upstream/` into
`upstream/redraw-subset.tar.gz` as well; the archive is rebuilt byte-identically
from an unchanged subset, so it re-uploads only when its contents move. Publish
it deliberately: the subset carries World Values Survey data.

```bash
uv run --no-sync python -m machine_bias_reproduction hub push --upstream --dry-run
uv run --no-sync python -m machine_bias_reproduction hub push --upstream
uv run --no-sync python -m machine_bias_reproduction hub push --upstream-only   # the archive alone, nothing under data/
```

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
