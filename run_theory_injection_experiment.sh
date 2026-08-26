#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

usage() {
    cat <<'EOF'
run_theory_injection_experiment.sh — Context/theory injection experiment driver.

Usage:
  ./run_theory_injection_experiment.sh smoke
  ./run_theory_injection_experiment.sh audit
  ./run_theory_injection_experiment.sh export
  ./run_theory_injection_experiment.sh plan
  ./run_theory_injection_experiment.sh run --model PATH [context_injection.run options...]
  ./run_theory_injection_experiment.sh compare
  ./run_theory_injection_experiment.sh all --model PATH [context_injection.run options...]
  ./run_theory_injection_experiment.sh check

Defaults target the reduced d_trust smoke test:
  QUESTION=d_trust
  MODE=ntp
  RAW_VARS=d_happy,d_religiousp,d_polpos
  SELECTED_VARS=d_happy,d_religiousp,d_polpos
  THEORY_MAP=src/context_injection/theories/reduced_proxy_theory_map.json
  PROMPTS=prompts/reduced_d_trust_ntp.jsonl
  RUN_NAME=reduced_d_trust

Override any default with an environment variable:
  QUESTION=d_happy PROMPTS=prompts/d_happy_ntp.jsonl ./run_theory_injection_experiment.sh export

`smoke` runs audit + export + dry-run plan. It does not need a model.
`run` and `all` need a GGUF model path through --model PATH or MODEL=PATH.
EOF
}

command_name="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi

QUESTION="${QUESTION:-d_trust}"
MODE="${MODE:-ntp}"
RAW_VARS="${RAW_VARS:-d_happy,d_religiousp,d_polpos}"
SELECTED_VARS="${SELECTED_VARS:-d_happy,d_religiousp,d_polpos}"
THEORY_MAP="${THEORY_MAP:-src/context_injection/theories/reduced_proxy_theory_map.json}"
PROMPTS="${PROMPTS:-prompts/reduced_d_trust_ntp.jsonl}"
RUN_NAME="${RUN_NAME:-reduced_d_trust}"
AUDIT_OUT="${AUDIT_OUT:-prompts/reduced_d_trust_mapping_audit.json}"
COMPARE_OUT="${COMPARE_OUT:-prompts/reduced_d_trust_condition_compare}"
MODEL="${MODEL:-}"

context_python() {
    require_uv
    uv run --locked python -m "$@"
}

has_model_arg() {
    local previous=""
    for argument in "$@"; do
        [[ "$argument" == "--model" || "$argument" == --model=* ]] && return 0
        [[ "$previous" == "--model" ]] && return 0
        previous="$argument"
    done
    return 1
}

audit_mapping() {
    context_python context_injection.audit \
        --theory-map "$THEORY_MAP" \
        --question "$QUESTION" \
        --out "$AUDIT_OUT"
}

export_prompts() {
    context_python context_injection \
        --out "$PROMPTS" \
        --question "$QUESTION" \
        --mode "$MODE" \
        --raw-vars "$RAW_VARS" \
        --selected-vars "$SELECTED_VARS" \
        --theory-map "$THEORY_MAP"
}

plan_run() {
    context_python context_injection.run \
        --prompts "$PROMPTS" \
        --run-name "$RUN_NAME" \
        --dry-run
}

run_conditions() {
    local args=(--prompts "$PROMPTS" --run-name "$RUN_NAME" --use-archived-fa --analysis)
    if [[ -n "$MODEL" ]]; then
        args+=(--model "$MODEL")
    elif ! has_model_arg "$@"; then
        die "run needs --model PATH or MODEL=PATH. Use 'plan' or 'smoke' for no-model checks."
    fi
    context_python context_injection.run "${args[@]}" "$@"
}

compare_conditions() {
    context_python context_injection.compare \
        --condition "C0=outputs/culture/context_${RUN_NAME}/C0/${QUESTION}" \
        --condition "C1=outputs/culture/context_${RUN_NAME}/C1/${QUESTION}" \
        --condition "C2=outputs/culture/context_${RUN_NAME}/C2/${QUESTION}" \
        --condition "C3=outputs/culture/context_${RUN_NAME}/C3/${QUESTION}" \
        --condition "C4=outputs/culture/context_${RUN_NAME}/C4/${QUESTION}" \
        --out "$COMPARE_OUT"
}

run_checks() {
    context_python context_injection._self_check
    require_uv
    uv run --locked python -m compileall -q src/context_injection
}

case "$command_name" in
    -h|--help|help) usage ;;
    audit) audit_mapping ;;
    export) export_prompts ;;
    plan) plan_run ;;
    smoke)
        audit_mapping
        export_prompts
        plan_run
        ;;
    run) run_conditions "$@" ;;
    compare) compare_conditions ;;
    all)
        audit_mapping
        export_prompts
        run_conditions "$@"
        compare_conditions
        ;;
    check) run_checks ;;
    *) echo "Unknown command: $command_name" >&2; usage >&2; exit 2 ;;
esac
