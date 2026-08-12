#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
14_culture_summary.sh — Build the cross-question culture summary.

Usage:
  ./scripts/14_culture_summary.sh [--models KEY...] [--cultures NAME...]
                                  [--questions Q...]

Every other culture figure answers one question at a time. This one reads across
happiness, politics and religion together, because the three claims the culture
extension rests on are about the pattern over outcomes rather than any single
one of them:

  fig_overall_nemd     how far each model sits from the survey
  fig_compression      how much it flattens the differences between groups
  fig_home_advantage   whether it is closer to its own culture than the
                       reference is, on identical subpopulations
  fig_sweep_cost       wall clock per finished run

Writes them to figures/culture/<model>/summary/ with culture_summary.csv,
culture_home_advantage.csv and sweep_cost.csv beside them in
outputs/culture/<model>/summary/, carrying exactly the plotted numbers.

The home-advantage table breaks english into United States and Australia as well
as the pooled pair, so a pooled result cannot hide two countries pulling in
opposite directions.

Runs no inference and loads no model, so it is safe to call while a sweep is
still in progress; a question with no finished run contributes no bar.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python culture-summary "$@"
