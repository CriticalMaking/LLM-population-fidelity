#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
01_archived.sh — Reproduce Mixtral results from archived NTP and FA outputs.

Usage:
  ./scripts/01_archived.sh [--questions Q...] [--force]

Questions: d_happy d_polpos d_religiousp d_trust (or 'all'; default d_happy)

Validates the canonical inputs, regenerates all statistical tables and figures,
and verifies the numeric checkpoints reported by the paper. The paper publishes
those checkpoints for happiness only, so they are asserted for d_happy alone.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python archived "$@"
