#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
02_fresh.sh — Run resumable fresh Mixtral happiness inference.

Usage:
  ./scripts/02_fresh.sh --model PATH [--mode ntp|fa|all] [--limit N] [--force]
                        [--threads N] [--gpu-layers N] [--legacy-unseeded-fa]

The full run requires the pinned 26.4 GB Q4_K_M GGUF. A limited run is an
inference smoke test and intentionally skips the complete statistical analysis.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv
uv run --locked --extra inference python -m machine_bias_reproduction fresh "$@"
