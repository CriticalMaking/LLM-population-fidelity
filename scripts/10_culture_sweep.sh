#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
10_culture_sweep.sh — Run the culture experiment across many model/culture pairs.

Usage:
  ./scripts/10_culture_sweep.sh [--models all|KEY...] [--cultures all|NAME...]
                                [--questions all|Q...]
                                [--redo] [--no-compare] [--dry-run]
                                [run options...]

Runs one invocation per model/culture pair instead of one invocation for the
whole grid, so a pair that fails or is interrupted never takes the rest of the
sweep with it. Each pair writes its own log, already-complete pairs are skipped
unless --redo is given, and the cross-culture comparison is rebuilt at the end.

Examples:
  ./scripts/10_culture_sweep.sh                                   # every pair
  ./scripts/10_culture_sweep.sh --cultures portuguese             # one culture, both models
  ./scripts/10_culture_sweep.sh --models gemma4_31b --cultures all
  ./scripts/10_culture_sweep.sh --cultures english german --mode ntp
  ./scripts/10_culture_sweep.sh --questions all --models gemma4_31b

Any option this script does not recognise is passed through to the underlying
run, so --mode, --batch-size, --fa-max-tokens, --force and --legacy-unseeded-fa
all work here too.
EOF
}

ALL_MODELS=(gemma4_31b qwen3_vl_8b)
ALL_CULTURES=(arabic bengali chinese english german korean portuguese spanish turkish)
ALL_QUESTIONS=(d_happy d_polpos d_religiousp d_trust)

models=()
cultures=()
questions=()
passthrough=()
skip_completed=1
run_compare=1
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do models+=("$1"); shift; done
            ;;
        --cultures)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do cultures+=("$1"); shift; done
            ;;
        --questions)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do questions+=("$1"); shift; done
            ;;
        --redo) skip_completed=0; shift ;;
        --no-compare) run_compare=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

if [[ ${#models[@]} -eq 0 || "${models[0]}" == "all" ]]; then
    models=("${ALL_MODELS[@]}")
fi
if [[ ${#cultures[@]} -eq 0 || "${cultures[0]}" == "all" ]]; then
    cultures=("${ALL_CULTURES[@]}")
fi
if [[ ${#questions[@]} -eq 0 ]]; then
    questions=(d_happy)
elif [[ "${questions[0]}" == "all" ]]; then
    questions=("${ALL_QUESTIONS[@]}")
fi

for model in "${models[@]}"; do
    printf '%s\n' "${ALL_MODELS[@]}" | grep -qx -- "$model" ||
        die "unknown model: $model (known: ${ALL_MODELS[*]})"
done
for culture in "${cultures[@]}"; do
    printf '%s\n' "${ALL_CULTURES[@]}" | grep -qx -- "$culture" ||
        die "unknown culture: $culture (known: ${ALL_CULTURES[*]})"
done
for question in "${questions[@]}"; do
    printf '%s\n' "${ALL_QUESTIONS[@]}" | grep -qx -- "$question" ||
        die "unknown question: $question (known: ${ALL_QUESTIONS[*]})"
done

LOG_DIR="$REPO_ROOT/outputs/culture/logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/sweep_summary.tsv"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

total=$(( ${#models[@]} * ${#cultures[@]} * ${#questions[@]} ))
echo "Culture sweep: ${#models[@]} model(s) x ${#cultures[@]} culture(s) x ${#questions[@]} question(s) = $total run(s)"
echo "Models:    ${models[*]}"
echo "Cultures:  ${cultures[*]}"
echo "Questions: ${questions[*]}"
[[ ${#passthrough[@]} -gt 0 ]] && echo "Run options: ${passthrough[*]}"
echo "Logs:     ${LOG_DIR#"$REPO_ROOT"/}"
echo

if [[ $dry_run -eq 1 ]]; then
    for question in "${questions[@]}"; do
        for model in "${models[@]}"; do
            for culture in "${cultures[@]}"; do
                echo "would run: --models $model --cultures $culture --questions $question ${passthrough[*]}"
            done
        done
    done
    exit 0
fi

printf 'model\tculture\tquestion\tstatus\tseconds\tvalid_answer_mass\tvalid\tprompts\tvalid_rate\n' > "$SUMMARY"

index=0
completed=0
skipped=0
failed=0

for question in "${questions[@]}"; do
  for model in "${models[@]}"; do
    for culture in "${cultures[@]}"; do
        index=$((index + 1))
        pair="$model/$culture/$question"
        outputs="$REPO_ROOT/outputs/culture/$model/$culture/$question"
        log="$LOG_DIR/$model-$culture-$question.log"

        # capacity.csv, not summary_metrics.csv: a run whose finetuned culture
        # MLLM could not answer the paper's prompt writes no distances but is
        # still complete.
        if [[ $skip_completed -eq 1 && -f "$outputs/capacity.csv" ]]; then
            echo "[$index/$total] $pair — already run, skipping (--redo to rerun)"
            printf '%s\t%s\t%s\tskipped\t0\t\t\t\t\n' "$model" "$culture" "$question" >> "$SUMMARY"
            skipped=$((skipped + 1))
            continue
        fi

        echo "[$index/$total] $pair — running (log: ${log#"$REPO_ROOT"/})"
        pair_started=$(date +%s)
        if "$REPO_ROOT/scripts/07_culture.sh" \
                --models "$model" \
                --cultures "$culture" \
                --questions "$question" \
                --skip-compare \
                "${passthrough[@]}" > "$log" 2>&1; then
            status="ok"
            completed=$((completed + 1))
        else
            status="FAILED"
            failed=$((failed + 1))
        fi
        elapsed=$(( $(date +%s) - pair_started ))
        mass="$(grep -o 'mean valid-answer mass [0-9.]*' "$log" | tail -1 | awk '{print $NF}')"
        capacity="$(grep -o 'capacity: [0-9]*/[0-9]* prompts answered in the paper.s format ([0-9.]*%)' "$log" | tail -1)"
        valid="$(printf '%s' "$capacity" | sed -n 's|.*capacity: \([0-9]*\)/.*|\1|p')"
        prompts="$(printf '%s' "$capacity" | sed -n 's|.*capacity: [0-9]*/\([0-9]*\) .*|\1|p')"
        rate="$(printf '%s' "$capacity" | sed -n 's|.*(\([0-9.]*\)%).*|\1|p')"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$model" "$culture" "$question" "$status" "$elapsed" "$mass" "$valid" "$prompts" "$rate" >> "$SUMMARY"
        echo "    $status in ${elapsed}s${mass:+, valid-answer mass $mass}${rate:+, answered ${rate}%}"
        if [[ "$status" == "FAILED" ]]; then
            echo "    last lines of $log:"
            tail -5 "$log" | sed 's/^/      /'
        fi
    done
  done
done

echo
if [[ $run_compare -eq 1 && $completed -gt 0 ]]; then
    echo "Rebuilding cross-culture comparison..."
    "$REPO_ROOT/scripts/09_culture_compare.sh" \
        --models "${models[@]}" --questions "${questions[@]}" || true
    echo
fi

echo "Sweep started $started_at, finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  completed: $completed   skipped: $skipped   failed: $failed"
echo "  summary:   ${SUMMARY#"$REPO_ROOT"/}"
column -t -s $'\t' "$SUMMARY" | sed 's/^/  /'

[[ $failed -eq 0 ]]
