#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

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

No arguments run the archived happiness analysis: regenerate every table and
figure and assert the paper's numeric checkpoints, published for happiness
alone. fresh needs the pinned 26.4 GB GGUF; with --limit it is a smoke test.

compare tests archived vs fresh for equivalence on paired subpopulation nEMD
differences (±0.005; --equivalence-margin to override). trace audits the fresh
run's per-inference records, exiting non-zero if any is untraceable. verify
hashes all consumed upstream inputs; --full also CRC-checks the ZIP.

The paper's own tables and figures: ./run_reports.sh. The culture extension:
./run_experiment_add_culture.sh.
EOF
}

command_name="${1:-archived}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    -h|--help|help) usage ;;
    archived) run_python archived "$@" ;;
    fresh) run_python --extra inference fresh "$@" ;;
    compare) require_uv; uv run --locked python -m machine_bias_reproduction.comparison "$@" ;;
    trace) run_python trace "$@" ;;
    verify) run_python verify "$@" ;;
    test) require_uv; uv run --locked pytest tests "$@" ;;
    lint)
        require_uv
        uv run --locked ruff format --check src tests
        uv run --locked ruff check src tests
        uv run --locked mypy
        ;;
    *) echo "Unknown command: $command_name" >&2; usage >&2; exit 2 ;;
esac
