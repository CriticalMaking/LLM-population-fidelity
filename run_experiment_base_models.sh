#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
run_experiment_base_models.sh — The un-finetuned baseline for the culture
experiment.

Usage:
  ./run_experiment_base_models.sh run [--models all|KEY...]
                                      [--questions all|Q...]
                                      [--first-countries NAME...] [--no-priority]
                                      [--redo] [--no-compare] [--dry-run]
                                      [--allow-concurrent]
  ./run_experiment_base_models.sh compare [--models KEY...] [--questions Q...]
  ./run_experiment_base_models.sh summary [--models KEY...] [--questions Q...]

Models:    gemma4_31b, gemma4_e4b, qwen3_vl_8b
           (default gemma4_31b; --models all for every one)
Questions: d_happy d_polpos d_religiousp d_trust (default: all four)

Runs each base model exactly as ./run_experiment_add_culture.sh runs the
culture-finetuned LLMs, and exactly as ./run_experiment.sh fresh ran Mixtral:
the paper's own prompts, the same batched NTP and FA code, the same capacity
measurement, the same analysis.

Why it exists: every distance the culture experiment reports is currently read
against Mixtral, which is a *different* base model, so the comparison confounds
the culture finetuning with the base model it was fitted on. The base arm is the
same weights before finetuning, which is the only reference that isolates what
the finetuning did.

Results land under outputs/culture/<model>/base/<question>/, beside the
culture-finetuned runs, so `compare` and `summary` pick the base up as another
arm. Two artifacts exist only once it has run:

  base_deltas.csv           per country, each arm's distance minus the base's on
                            the same subpopulations, with the nEMD_center and
                            country coefficients paired the same way
  fig_home_advantage_base   the home-advantage difference-in-differences taken
                            against the model's own base rather than Mixtral

Germany's prompts are generated first by default. That is ordering, not
selection: a finished run is identical either way, but a run still in progress
already holds the country the German comparison needs. Pass --no-priority for
the plain order, or --first-countries to name others.

Each run is 13,904 NTP + 26,981 FA prompts and resumable, so an interrupted run
continues where it stopped. Nothing else should be using the card: the driver
refuses to start while another culture run is in flight.

  ./run_experiment_base_models.sh run --dry-run          # what would run
  ./run_experiment_base_models.sh run --models all       # both base models
  ./run_experiment_base_models.sh summary                # rebuild the tables
EOF
}

command_name="${1:-run}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    -h|--help|help) usage ;;
    run) ./scripts/15_base_models.sh "$@" ;;
    compare) ./scripts/09_culture_compare.sh "$@" ;;
    summary) ./scripts/14_culture_summary.sh "$@" ;;
    *) echo "Unknown command: $command_name" >&2; usage >&2; exit 2 ;;
esac
