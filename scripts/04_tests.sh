#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
04_tests.sh — Run unit and archived integration tests.

Usage:
  ./scripts/04_tests.sh [pytest arguments...]

Examples:
  ./scripts/04_tests.sh
  ./scripts/04_tests.sh -m integration -vv
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv
uv run --locked pytest tests "$@"
