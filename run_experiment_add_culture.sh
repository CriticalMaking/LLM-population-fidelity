#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
run_experiment_add_culture.sh — Culture-finetuned MLLM extension of the
Machine Bias reproduction.

Usage:
  ./run_experiment_add_culture.sh adapters [--source DIR] [--models KEY...]
                                           [--cultures NAME...] [--force]
  ./run_experiment_add_culture.sh smoke [--models KEY...] [--cultures NAME...]
  ./run_experiment_add_culture.sh run [--models KEY...] [--cultures NAME...]
                                      [--questions Q...] [--mode ntp|fa|all]
                                      [--batch-size N] [--fa-max-tokens N]
                                      [--force] [--legacy-unseeded-fa]
                                      [--skip-compare]
  ./run_experiment_add_culture.sh sweep [--models all|KEY...]
                                        [--cultures all|NAME...]
                                        [--questions all|Q...]
                                        [--redo] [--no-compare] [--dry-run]
  ./run_experiment_add_culture.sh compare [--models KEY...] [--cultures NAME...]
                                          [--questions Q...]

Models:    gemma4_31b, qwen3_vl_8b
Cultures:  arabic bengali chinese english german korean portuguese spanish turkish
Questions: d_happy d_polpos d_religiousp d_trust (default d_happy)

Adapters answer the paper's own prompt, so their distances are comparable with
the archived Mixtral run. How often each adapter can honour that prompt is
measured per run in capacity.csv and charted in fig_culture_capacity.

Start with `adapters` to stage the LoRA weights under models/culture, then
`smoke` to measure throughput and the valid-answer probe before committing.

`sweep` is the usual way to run the experiment: it drives one pair at a time,
so a failing adapter never takes the rest of the grid with it, writes a log per
pair, skips pairs already analysed, and rebuilds the comparison at the end.

  ./run_experiment_add_culture.sh sweep                          # all 18 pairs
  ./run_experiment_add_culture.sh sweep --cultures portuguese    # one culture
  ./run_experiment_add_culture.sh sweep --models gemma4_31b --cultures all

`run` is the single-invocation form for one pair or a small selection. Both are
resumable: every prompt writes its own atomic result file, so an interrupted
sweep continues where it stopped rather than starting over. `compare` rebuilds
the cross-culture figures and reports from whatever has finished.
EOF
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    -h|--help|help) usage ;;
    adapters) ./scripts/08_culture_adapters.sh "$@" ;;
    smoke) ./scripts/07_culture.sh --limit 2 --skip-compare "$@" ;;
    run) ./scripts/07_culture.sh "$@" ;;
    sweep) ./scripts/10_culture_sweep.sh "$@" ;;
    compare) ./scripts/09_culture_compare.sh "$@" ;;
    *) echo "Unknown command: $command_name" >&2; usage >&2; exit 2 ;;
esac
