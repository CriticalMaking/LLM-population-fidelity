#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
run_experiment.sh — Mixtral Machine Bias reproduction driver.

Usage:
  ./run_experiment.sh
  ./run_experiment.sh archived [--questions Q...] [--force]
  ./run_experiment.sh fresh --model PATH [--question Q] [fresh options...]
  ./run_experiment.sh compare [comparison options...]
  ./run_experiment.sh trace [--question Q] [--mode ntp|fa|all]
  ./run_experiment.sh verify [--full]
  ./run_experiment.sh test [pytest arguments...]
  ./run_experiment.sh lint

Questions: d_happy d_polpos d_religiousp d_trust (or 'all'; default d_happy)

No arguments select the archived, fully reproducible happiness analysis. Fresh
inference is explicit because it requires the pinned 26.4 GB GGUF and
substantial compute.

The paper's own tables and figures are reproduced by ./run_reports.sh, and the
culture extension by ./run_experiment_add_culture.sh.
EOF
}

command_name="${1:-archived}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    -h|--help|help) usage ;;
    archived) ./scripts/01_archived.sh "$@" ;;
    fresh) ./scripts/02_fresh.sh "$@" ;;
    compare) ./scripts/11_compare.sh "$@" ;;
    trace) ./scripts/06_trace.sh "$@" ;;
    verify) ./scripts/03_verify.sh "$@" ;;
    test) ./scripts/04_tests.sh "$@" ;;
    lint) ./scripts/05_lint.sh "$@" ;;
    *) echo "Unknown command: $command_name" >&2; usage >&2; exit 2 ;;
esac
