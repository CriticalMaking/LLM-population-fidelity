#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

API_FA_MAX_TOKENS_DEFAULT=300

SERVED_MODELS=(luna terra sol)

is_served_model() {
    printf '%s\n' "${SERVED_MODELS[@]}" | grep -qx -- "$1"
}

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
    local collecting=0 named=0 has_muse=0 has_api=0 has_others=0 argument
    note_model() {
        named=1
        case "$1" in
            muse_glimmer_30b) has_muse=1 ;;
            all) has_muse=1; has_others=1 ;;
            *) if is_served_model "$1"; then has_api=1; else has_others=1; fi ;;
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
    if [[ $has_api -eq 1 && ( $has_muse -eq 1 || $has_others -eq 1 ) ]]; then
        die "the served models (${SERVED_MODELS[*]}) carry no torch stack, so they run apart from
the local ones: name only served models, or only local ones"
    fi
    if [[ $has_muse -eq 1 && $has_others -eq 1 ]]; then
        die "muse_glimmer_30b needs its own transformers pin (the muse extra) and must run alone:
  pass --models muse_glimmer_30b by itself, or name the other models without it"
    fi
    if [[ $has_api -eq 1 ]]; then
        echo api
    elif [[ $has_muse -eq 1 ]]; then
        echo muse
    else
        echo culture
    fi
}

run_culture() {
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    local extra
    extra="$(culture_extra_for "$@")"
    local arguments=("$@")
    if [[ "$extra" == "api" ]]; then
        local has_fa_max_tokens=0 argument
        for argument in "$@"; do
            [[ "$argument" == "--fa-max-tokens" || "$argument" == --fa-max-tokens=* ]] &&
                has_fa_max_tokens=1
        done
        if [[ $has_fa_max_tokens -eq 0 ]]; then
            echo "the served models reason before they answer: defaulting --fa-max-tokens" \
                "$API_FA_MAX_TOKENS_DEFAULT so hidden reasoning tokens don't starve the visible" \
                "answer (pass --fa-max-tokens yourself to override)" >&2
            arguments+=(--fa-max-tokens "$API_FA_MAX_TOKENS_DEFAULT")
        fi
    fi
    run_python --extra "$extra" culture "${arguments[@]}"
}

run_served_smoke() {
    run_python --extra api culture-served-smoke "$@"
}
