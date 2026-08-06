#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
12_reports.sh — reproduce the tables and figures published with the paper.

Usage:
  ./scripts/12_reports.sh [SECTION] [--questions Q...] [--series S...]

Sections: all, main, appendix, robustness, tables, figures, smoke

Reads only the archived upstream outputs, so it loads no model and runs no
inference. Writes CSV and booktabs LaTeX under outputs/reports and plates under
figures/reports.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python reports "$@"
