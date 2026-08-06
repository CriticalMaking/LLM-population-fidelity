#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
11_compare.sh — Statistically compare archived and fresh experiment results.

Usage:
  ./scripts/11_compare.sh [comparison options...]

Uses paired subpopulation nEMD differences, country-survey-wave cluster
randomization, Holm correction, and an equivalence test. Writes tables,
a comparison figure, and a hashed manifest to outputs/comparison.

The default practical-equivalence margin is ±0.005 nEMD. Override it with
--equivalence-margin VALUE when a different smallest important change is
scientifically justified.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv
uv run --locked python scripts/compare_runs.py "$@"
