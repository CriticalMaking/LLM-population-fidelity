#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
03_verify.sh — Verify canonical source artifacts and generated checkpoints.

Usage:
  ./scripts/03_verify.sh [--full]

--full additionally CRC-checks the ZIP and verifies every substantive extracted
file by path and size. The normal check hashes all consumed upstream inputs.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python verify "$@"
