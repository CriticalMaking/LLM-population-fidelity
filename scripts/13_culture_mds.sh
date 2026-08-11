#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
13_culture_mds.sh — Build the culture-matched MDS plate.

Usage:
  ./scripts/13_culture_mds.sh [--models KEY...] [--cultures NAME...]
                              [--questions Q...]

Embeds each culture's adapter and both Mixtral series — the archived run and the
fresh re-run — in one shared MDS space, restricted to the WVS subpopulations of
the countries where that culture's language is dominant. Adapter and references
are therefore scored on identical respondents, so a difference between the point
clouds is the model and not the sample.

Only english, german and spanish are drawn: the other six cultures have no WVS
respondent block. Writes fig_culture_mds_matched to figures/culture/<model>/<q>/
and the per-panel means to outputs/culture/<q>/culture_mds_matched.csv.

Runs no inference and loads no model, so it is safe to call while a sweep is
still in progress; a run whose consolidated CSVs are missing is skipped.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python culture-mds "$@"
