#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

usage() {
    cat <<'EOF'
run_experiment_add_culture.sh — Culture-finetuned MLLM extension of the
Machine Bias reproduction.

Usage:
  ./run_experiment_add_culture.sh adapters [--source DIR] [--models KEY...]
                                           [--cultures NAME...] [--force]
  ./run_experiment_add_culture.sh health [--models KEY...] [--cultures NAME...]
  ./run_experiment_add_culture.sh smoke --models KEY... [--cultures NAME...]
  ./run_experiment_add_culture.sh run --models KEY... [--cultures NAME...]
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
  ./run_experiment_add_culture.sh mds [--models KEY...] [--cultures NAME...]
                                      [--questions Q...] [--all-countries]
                                      [--outcomes]
  ./run_experiment_add_culture.sh summary [--models KEY...] [--cultures NAME...]
                                          [--questions Q...]

Models:    gemma4_31b, gemma4_e4b, qwen3_vl_8b, muse_glimmer_30b
           (muse_glimmer_30b runs alone — its transformers pin lives in the
           muse extra, selected automatically; german is its only adapter.
           run and smoke need --models named: the default set spans both pins)
Cultures:  arabic bengali chinese english german korean portuguese spanish turkish
           (sweep default german; --cultures all for the full grid)
Questions: d_happy d_polpos d_religiousp d_trust (default d_happy)

The finetuned culture MLLMs answer the paper's own prompt, so their distances
are comparable with the archived Mixtral run; how often each arm can honour
that prompt is measured per run in capacity.csv and fig_culture_capacity.
Per-mode figures come as separate _ntp and _fa files because the two modes
fail independently.

Start with `adapters` to stage the LoRA weights under models/culture, then
`smoke` for the throughput and valid-answer probe. `adapters` prints a health
line per staged adapter and `health` reprints them: a diverged checkpoint
costs an hour a question through inference, seconds through the weights.

`sweep` drives one model/culture/question at a time — a failing adapter never
takes the grid with it — with one log per run, finished runs skipped
(--redo to force), and the comparison rebuilt at the end:

  ./run_experiment_add_culture.sh sweep                # every model, german, d_happy
  ./run_experiment_add_culture.sh sweep --cultures portuguese    # one culture
  ./run_experiment_add_culture.sh sweep --models gemma4_31b --cultures all

`run` is the single-invocation form. Both are resumable: every prompt writes
its own atomic result file. `compare` rebuilds the per-question cross-culture
figures and tables, `mds` the shared-space plates, and `summary` the three
cross-question charts, each with the CSV of its numbers beside it.
EOF
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    -h|--help|help) usage ;;
    adapters) run_python culture-adapters "$@" ;;
    health) run_python culture-health "$@" ;;
    smoke) run_culture --limit 2 --skip-compare "$@" ;;
    run) run_culture "$@" ;;
    sweep) ./scripts/sweep.sh "$@" ;;
    compare) run_python culture-compare "$@" ;;
    mds) run_python culture-mds "$@" ;;
    summary) run_python culture-summary "$@" ;;
    *) echo "Unknown command: $command_name" >&2; usage >&2; exit 2 ;;
esac
