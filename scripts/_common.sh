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

culture_extra_for() {
    local collecting=0 named=0 has_muse=0 has_others=0 argument
    note_model() {
        named=1
        case "$1" in
            muse_glimmer_30b) has_muse=1 ;;
            all) has_muse=1; has_others=1 ;;
            *) has_others=1 ;;
        esac
    }
    for argument in "$@"; do
        case "$argument" in
            --models) collecting=1; continue ;;
            --models=*) note_model "${argument#--models=}"; collecting=0; continue ;;
            --*) collecting=0; continue ;;
        esac
        [[ $collecting -eq 1 ]] && note_model "$argument"
    done
    if [[ $named -eq 0 ]]; then
        die "name --models explicitly: the default model set spans the culture and muse extras"
    fi
    if [[ $has_muse -eq 1 && $has_others -eq 1 ]]; then
        die "muse_glimmer_30b needs its own transformers pin (the muse extra) and must run alone:
  pass --models muse_glimmer_30b by itself, or name the other models without it"
    fi
    if [[ $has_muse -eq 1 ]]; then
        echo muse
    else
        echo culture
    fi
}

run_culture() {
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    local extra
    extra="$(culture_extra_for "$@")"
    run_python --extra "$extra" culture "$@"
}
