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
    require_uv
    uv run --locked python -m machine_bias_reproduction "$@"
}
