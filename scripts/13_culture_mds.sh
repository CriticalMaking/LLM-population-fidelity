#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
13_culture_mds.sh — Build the culture MDS plates.

Usage:
  ./scripts/13_culture_mds.sh [--models KEY...] [--cultures NAME...]
                              [--questions Q...] [--all-countries] [--outcomes]

Embeds each finetuned culture MLLM and both Mixtral series — the archived run
and the fresh re-run — in one shared MDS space, restricted to the WVS
subpopulations of the countries where that culture's language is dominant. Model
and references are therefore scored on identical respondents, so a difference
between the point clouds is the model and not the sample.

  (default)         fig_culture_mds_matched — own countries only
  --all-countries   fig_culture_mds_all_countries — every WVS country, with the
                    matched subset ringed, so the default plate's sample is
                    visible against the whole survey
  --outcomes        fig_culture_mds_outcomes — one culture across happiness,
                    politics and religion, drawn once per culture

Only english, german and spanish are drawn: the other six cultures have no WVS
respondent block. Figures land in figures/culture/<model>/<culture>/<q>/ and the
per-panel means in outputs/culture/<q>/culture_mds_matched.csv.

Runs no inference and loads no model, so it is safe to call while a sweep is
still in progress; a run whose consolidated CSVs are missing is skipped.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python culture-mds "$@"
