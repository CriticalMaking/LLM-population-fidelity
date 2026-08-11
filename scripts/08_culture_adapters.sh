#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
08_culture_adapters.sh — Stage trained culture LoRA adapters under models/culture.

Usage:
  ./scripts/08_culture_adapters.sh [--source DIR] [--models KEY...]
                                   [--cultures NAME...] [--force]

Copies only the end-of-training adapter of each <culture>/<model>/cultural
directory: the per-step checkpoint-N directories are optimizer state, not the
selected model, and are excluded. Writes models/culture/ADAPTERS.json with the
SHA-256 of every adapter, its declared base model, and its LoRA configuration.

A repeat run re-hashes what is already staged, so a truncated earlier copy is
replaced rather than trusted.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python culture-adapters "$@"
