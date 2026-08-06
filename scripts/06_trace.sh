#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
06_trace.sh — Audit the per-inference traces of the fresh run.

Usage:
  ./scripts/06_trace.sh [--mode ntp|fa|all]

Rebuilds every prompt, matches it against the stored per-inference record, and
verifies that each record carries a complete trace: run id, model SHA-256,
backend build, sampling parameters, seed, timing, and prompt hash. Writes
outputs/fresh/inference_trace.csv and inference_trace_summary.json, and exits
non-zero when any record is untraceable or failed.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python trace "$@"
