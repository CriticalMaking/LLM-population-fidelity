#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_uv() {
    command -v uv >/dev/null 2>&1 || die "uv is required: https://docs.astral.sh/uv/"
}

run_python() {
    local extra=()
    if [[ "${1:-}" == "--extra" ]]; then
        extra=(--extra "$2")
        shift 2
    fi
    require_uv
    uv run --locked "${extra[@]}" python -m machine_bias_reproduction "$@"
}

run_culture() {
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    run_python --extra culture culture "$@"
}
