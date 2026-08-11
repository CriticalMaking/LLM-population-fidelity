#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
09_culture_compare.sh — Build cross-culture comparison figures and reports.

Usage:
  ./scripts/09_culture_compare.sh [--models KEY...] [--cultures NAME...]
                                  [--questions Q...]

Reads the per-run tables already written under outputs/culture and emits the
answer-format capacity panel, the nEMD density overlay, the per-distance
density grid, quality-band bars, culture ranking, culture-by-country heatmap,
answer-distribution panels and the culture-matched subset figures, plus
culture_comparison.csv, culture_matched_comparison.csv and REPORT.md for each
model and for the two models together.

Runs no inference and loads no model, so it is safe to call at any point while
a sweep is still in progress; incomplete runs are simply skipped.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python culture-compare "$@"
