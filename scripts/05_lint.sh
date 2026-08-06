#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
05_lint.sh — Run non-mutating style and type checks.

Usage:
  ./scripts/05_lint.sh

Checks Ruff formatting, Ruff lint rules, and strict mypy without rewriting files.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv
uv run --locked ruff format --check src tests
uv run --locked ruff check src tests
uv run --locked mypy
