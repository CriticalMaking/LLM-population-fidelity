# Context Injection

Adds context-injected prompt runs to the Machine Bias / WVS pipeline.

This package does not replace `machine_bias_reproduction`. It builds prompts
with extra respondent context, writes them as JSONL, and sends them back through
the existing inference and analysis code.

Full handoff notes live here:

```text
context/context_injection_0825.md
```

```text
src/context_injection/
├── base.py                         ContextSpec and ContextPayload
├── conditions.py                   C0-C4 condition definitions
├── strategies.py                   raw, semantic, selected-vars, structured context
├── prompts.py                      wraps the original Machine Bias prompt builder
├── export.py                       writes context prompt JSONL + manifest
├── records.py                      loads JSONL back into PromptRecord objects
├── run.py                          runs each condition through old inference paths
├── metadata.py                     writes a WVS variable inventory
├── audit.py                        checks a theory mapping against WVS data
├── compare.py                      stacks per-condition analysis outputs
└── theories/
    └── inglehart_welzel.py         mapping loader and construct scoring
```

## Contents

- [Goal](#goal)
- [What Exists](#what-exists)
- [Install And Checks](#install-and-checks)
- [Data Config](#data-config)
- [Conditions](#conditions)
- [Reduced CSV Smoke Test](#reduced-csv-smoke-test)
- [Run The Conditions](#run-the-conditions)
- [Full WVS Theory Test](#full-wvs-theory-test)
- [Theory Mapping](#theory-mapping)
- [GPT And Claude](#gpt-and-claude)
- [Open Work](#open-work)

## Goal

Test whether context built from social theory helps a model match WVS opinion
distributions better than the baseline prompt or a larger set of raw facts.

The comparison ladder is:

| Condition | Meaning | What changes |
| --- | --- | --- |
| `C0` | Baseline | Original Machine Bias prompt, no extra context |
| `C1` | Raw context | Adds selected respondent variables as plain facts |
| `C2` | Semantic control | Same variables as `C1`, written as a short sentence |
| `C3` | Selected-vars | Selected variables shown as plain facts |
| `C4` | Theory-structured | Same variables as `C3`, scored into theory dimensions |

The main comparisons are:

| Comparison | Question |
| --- | --- |
| `C1` vs `C0` | Does more context help? |
| `C2` vs `C1` | Does a simple summary help? |
| `C3` vs `C1` | Does theory-based variable choice help? |
| `C4` vs `C3` | Does theory structure help beyond variable choice? |
| `C4` vs `C2` | Does formal theory beat a generic summary? |

## What Exists

Working now:

- C0-C4 prompt construction.
- Exact target-variable exclusion.
- Root config for the sibling Machine Bias data directory.
- WVS metadata inventory command.
- Prompt JSONL export with the context details attached.
- Theory mapping JSON loader and scorer.
- Mapping audit command.
- Reduced CSV smoke-test mapping.
- Condition runner that writes one output directory per condition.
- Condition comparison stacker for `summary_metrics.csv` and
  `subpopulation_distances.csv`.

Not done yet:

- Full WVS questionnaire extract.
- Verified theory variables and scoring rules.
- Checks for target-like variables, not just exact target names.
- GPT/Claude API backend.
- Final paper-style condition plots and tables.

## Install And Checks

Use the repo setup from the root README.

Basic checks:

```bash
uv run python -m context_injection._self_check
uv run python -m compileall -q src/context_injection
```

Style sanity check used during this work:

```bash
awk 'length($0) > 100 { print FILENAME ":" FNR ":" length($0) ":" $0 }' \
  src/context_injection/*.py src/context_injection/theories/*.py
```

Pandas or Arrow may print sandbox CPU warnings. They are fine if the command
exits with code `0`.

## Data Config

The root file is:

```text
context_injection_config.json
```

Current values point to:

```text
../Machine-Bias-replication/data
```

Important entries:

| Key | Meaning |
| --- | --- |
| `wvs_csv` | WVS rows used for prompt export |
| `wvs_questions_csv` | question text and metadata |
| `wvs_levels_csv` | value labels |
| `subpops_csv` | subpopulation labels used by old analysis |
| `llm_outputs_csv_root` | archived model CSVs |

The current `wvs_csv` is the reduced Machine Bias CSV. It is enough for smoke
testing. It is not enough for the real theory experiment because it does not
contain the full WVS questionnaire.

Generate the current inventory:

```bash
uv run python -m context_injection.metadata \
  --out context/wvs_variable_inventory.md
```

## Conditions

The condition rules live in `conditions.py`.

| Condition | Source variables |
| --- | --- |
| `C0` | none |
| `C1` | `--raw-vars` |
| `C2` | `--raw-vars` |
| `C3` | `--selected-vars` |
| `C4` | `--selected-vars` |

Hard checks:

```text
C0 must add no context.
C1 and C2 must use the same variables.
C3 and C4 must use the same variables.
The target question must not appear in injected context.
```

Context is inserted just before the target question in the original prompt.
If a condition has no context text, the prompt stays unchanged.

## Reduced CSV Smoke Test

This only tests mechanics. It is not a real social-theory experiment.

For `d_trust`, the reduced CSV has only three useful non-target outcome
columns:

```text
d_happy
d_religiousp
d_polpos
```

The temporary mapping is:

```text
src/context_injection/theories/reduced_proxy_theory_map.json
```

The scores in that file are hand-coded ordinal proxy scores. They are not a
real Inglehart-Welzel mapping.

Audit the proxy mapping:

```bash
uv run python -m context_injection.audit \
  --theory-map src/context_injection/theories/reduced_proxy_theory_map.json \
  --question d_trust \
  --out prompts/reduced_d_trust_mapping_audit.json
```

Export reduced NTP prompts:

```bash
uv run python -m context_injection \
  --out prompts/reduced_d_trust_ntp.jsonl \
  --question d_trust \
  --mode ntp \
  --raw-vars d_happy,d_religiousp,d_polpos \
  --selected-vars d_happy,d_religiousp,d_polpos \
  --theory-map src/context_injection/theories/reduced_proxy_theory_map.json
```

Expected output for the current reduced file:

| Item | Count |
| --- | ---: |
| total prompt records | 69,520 |
| records per condition | 13,904 |
| conditions | `C0`, `C1`, `C2`, `C3`, `C4` |

The same audit, export, and dry-run plan are wrapped by the top-level script:

```bash
./run_theory_injection_experiment.sh smoke
```

The wrapper keeps this package wired into the repo-level workflow. It uses the
same `uv run --locked` style as `run_experiment.sh` and writes prompt artifacts
under `prompts/`.

## Run The Conditions

First check the output plan:

```bash
uv run python -m context_injection.run \
  --prompts prompts/reduced_d_trust_ntp.jsonl \
  --run-name reduced_d_trust \
  --dry-run
```

Expected output roots:

```text
outputs/culture/context_reduced_d_trust/C0/d_trust
outputs/culture/context_reduced_d_trust/C1/d_trust
outputs/culture/context_reduced_d_trust/C2/d_trust
outputs/culture/context_reduced_d_trust/C3/d_trust
outputs/culture/context_reduced_d_trust/C4/d_trust
```

Run one condition first:

```bash
uv run python -m context_injection.run \
  --prompts prompts/reduced_d_trust_ntp.jsonl \
  --run-name reduced_d_trust \
  --conditions C0 \
  --model models/Mixtral-8x7B-v0.1.Q4_K_M.gguf \
  --use-archived-fa \
  --analysis
```

Run all conditions after `C0` works:

```bash
uv run python -m context_injection.run \
  --prompts prompts/reduced_d_trust_ntp.jsonl \
  --run-name reduced_d_trust \
  --model models/Mixtral-8x7B-v0.1.Q4_K_M.gguf \
  --use-archived-fa \
  --analysis
```

Equivalent wrapper command:

```bash
./run_theory_injection_experiment.sh run \
  --model models/Mixtral-8x7B-v0.1.Q4_K_M.gguf
```

`--use-archived-fa` copies the archived FA CSV into each condition output so
the old analysis can run. That means this is an NTP context smoke test. It is
not a full FA context experiment.

Stack condition outputs after every condition has analysis files:

```bash
uv run python -m context_injection.compare \
  --condition C0=outputs/culture/context_reduced_d_trust/C0/d_trust \
  --condition C1=outputs/culture/context_reduced_d_trust/C1/d_trust \
  --condition C2=outputs/culture/context_reduced_d_trust/C2/d_trust \
  --condition C3=outputs/culture/context_reduced_d_trust/C3/d_trust \
  --condition C4=outputs/culture/context_reduced_d_trust/C4/d_trust \
  --out prompts/reduced_d_trust_condition_compare
```

Equivalent wrapper command:

```bash
./run_theory_injection_experiment.sh compare
```

`compare.py` currently stacks:

```text
summary_metrics.csv
subpopulation_distances.csv
```

## Full WVS Theory Test

The real experiment needs a full WVS respondent-level file aligned to the
Machine Bias rows.

Required columns:

```text
id
profile
i_surveyyear
i_country
i_age
i_sex
i_education
i_employment
i_marstat
d_happy
d_trust
d_religiousp
d_polpos
theory candidate variables
```

Steps:

1. Add full WVS data, question metadata, and level labels.
2. Update `context_injection_config.json`.
3. Generate a full inventory.
4. Pick broad `C1/C2` raw-context variables.
5. Pick `C3/C4` theory variables.
6. Write `context/verified_inglehart_welzel.json`.
7. Audit the mapping for each target question.
8. Export full prompts.
9. Run the condition runner.
10. Stack condition outputs.

Full inventory:

```bash
uv run python -m context_injection.metadata \
  --out context/full_wvs_variable_inventory.md
```

Audit a real mapping:

```bash
uv run python -m context_injection.audit \
  --theory-map context/verified_inglehart_welzel.json \
  --question d_trust \
  --out prompts/d_trust_mapping_audit.json
```

Export full prompts:

```bash
uv run python -m context_injection \
  --out prompts/full_d_trust_ntp.jsonl \
  --question d_trust \
  --mode ntp \
  --raw-vars <C1_VARIABLES> \
  --selected-vars <C3_C4_VARIABLES> \
  --theory-map context/verified_inglehart_welzel.json
```

Acceptance checks:

- WVS rows align with `subpops.csv` by `id`.
- Baseline variables are present.
- Target variables are present.
- Mapped theory variables are present.
- No target leakage.
- No unscored observed values.
- `C1/C2` variables match.
- `C3/C4` variables match.
- Sample prompts from all five conditions have been read by a person.

## Theory Mapping

Theory maps are JSON files. Each item says how one WVS answer should score on
one theory dimension.

Example shape:

```json
{
  "items": [
    {
      "variable": "actual_wvs_column",
      "dimension": "traditional_secular",
      "scores": {
        "answer label A": -1,
        "answer label B": 0,
        "answer label C": 1
      }
    }
  ]
}
```

`C4` averages scored items inside each dimension. Labels use simple thresholds:

| Score | Label type |
| ---: | --- |
| `<= -0.33` | low / traditional / survival |
| `>= 0.33` | high / secular-rational / self-expression |
| otherwise | mixed |

For a real run, the mapping needs notes outside the JSON:

```text
context/theory_variable_selection_notes.md
```

That note should record the WVS variable, question text, answer labels, scoring
direction, and reason for using it.

## GPT And Claude

There is no live GPT, Claude, OpenAI, or Anthropic backend in current HEAD.

Current GPT mentions are report labels for archived CSVs. They are not API
code.

If proprietary models stay in scope, start with FA:

```text
exported context prompts
-> records_by_condition()
-> API full-answer generation
-> raw records in the existing schema
-> consolidate FA CSV
-> run old analysis
```

NTP is not a small add-on for APIs. Local NTP uses answer-token probabilities.
Claude likely cannot provide the same thing. GPT depends on model and endpoint
logprob support. Decide the method before treating API NTP as comparable.

## Open Work

| Work | Why it matters |
| --- | --- |
| Run reduced `C0` live | Proves the runner works with the local model |
| Run reduced `C0-C4` live | Proves condition outputs and comparison work |
| Add full WVS extract | Required for real theory variables |
| Build verified mapping | Replaces the proxy smoke-test mapping |
| Add near-duplicate leakage checks | Avoids target leakage through related questions |
| Add API FA backend | Needed for GPT/Claude testing |
| Add final reports | Needed for paper-ready condition comparisons |
