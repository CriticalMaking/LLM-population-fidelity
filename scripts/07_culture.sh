#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
07_culture.sh — Run the culture-finetuned MLLM experiment.

Usage:
  ./scripts/07_culture.sh [--models KEY...] [--cultures NAME...]
                          [--questions Q...] [--mode ntp|fa|all] [--limit N]
                          [--batch-size N] [--fa-max-tokens N] [--force]
                          [--legacy-unseeded-fa] [--skip-compare]

Loads each base model once, attaches every selected culture LoRA, and runs
batched NTP and FA inference followed by capacity measurement, per-run analysis
and figures. Prompts are the paper's own, so the distances are comparable with
the archived Mixtral run; capacity.csv records how often each adapter could
answer in that format at all.

Results are per-prompt atomic files, so an interrupted run resumes where it
stopped. Stage the work with --models and --cultures; the full sweep is days of
GPU time.

Adapters must be staged first with ./scripts/08_culture_adapters.sh.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv
# Must be set before torch initialises CUDA, so it cannot move into Python.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
uv run --locked --extra culture python -m machine_bias_reproduction culture "$@"
