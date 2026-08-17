#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
15_base_models.sh — Run the un-finetuned base models on the paper's prompts.

Usage:
  ./scripts/15_base_models.sh [--models all|KEY...] [--questions all|Q...]
                              [--first-countries NAME...] [--no-priority]
                              [--redo] [--no-compare] [--dry-run]
                              [--allow-concurrent] [run options...]

The base arm is the same weights the culture-finetuned LLMs were fitted from,
sent the same prompts through the same code. It is the only reference that
isolates what the finetuning did: a gap against Mixtral confounds the finetuning
with the base model, because Mixtral is a different base model.

Results land beside the culture-finetuned runs, under
outputs/culture/<model>/base/<question>/, so every comparison figure and table
picks the base up as another arm without further work.

Germany's prompts are generated first by default. That is ordering, not
selection: a finished run is identical either way, but a run still in progress
already holds the country the German comparison needs.

Examples:
  ./scripts/15_base_models.sh                            # gemma4_31b, all four
  ./scripts/15_base_models.sh --models all
  ./scripts/15_base_models.sh --questions d_happy --dry-run
  ./scripts/15_base_models.sh --first-countries Germany Mexico

Any option this script does not recognise is passed through to the underlying
run, so --mode, --batch-size, --fa-max-tokens and --force all work here too.
EOF
}

ALL_MODELS=(gemma4_31b gemma4_e4b qwen3_vl_8b)
ALL_QUESTIONS=(d_happy d_polpos d_religiousp d_trust)
DEFAULT_MODELS=(gemma4_31b)
DEFAULT_FIRST_COUNTRIES=(Germany)

models=()
questions=()
first_countries=()
passthrough=()
no_priority=0
skip_completed=1
run_compare=1
dry_run=0
allow_concurrent=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do models+=("$1"); shift; done
            ;;
        --questions)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do questions+=("$1"); shift; done
            ;;
        --first-countries)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do first_countries+=("$1"); shift; done
            ;;
        --no-priority) no_priority=1; shift ;;
        --redo) skip_completed=0; shift ;;
        --no-compare) run_compare=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        --allow-concurrent) allow_concurrent=1; shift ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

if [[ ${#models[@]} -eq 0 ]]; then
    models=("${DEFAULT_MODELS[@]}")
elif [[ "${models[0]}" == "all" ]]; then
    models=("${ALL_MODELS[@]}")
fi
if [[ ${#questions[@]} -eq 0 || "${questions[0]}" == "all" ]]; then
    questions=("${ALL_QUESTIONS[@]}")
fi
if [[ ${#first_countries[@]} -eq 0 && $no_priority -eq 0 ]]; then
    first_countries=("${DEFAULT_FIRST_COUNTRIES[@]}")
fi
if [[ $no_priority -eq 1 ]]; then
    first_countries=()
fi

for model in "${models[@]}"; do
    printf '%s\n' "${ALL_MODELS[@]}" | grep -qx -- "$model" ||
        die "unknown model: $model (known: ${ALL_MODELS[*]})"
done
for question in "${questions[@]}"; do
    printf '%s\n' "${ALL_QUESTIONS[@]}" | grep -qx -- "$question" ||
        die "unknown question: $question (known: ${ALL_QUESTIONS[*]})"
done

if [[ $dry_run -eq 0 && $allow_concurrent -eq 0 ]]; then
    running="$(pgrep -af 'python -m machine_bias_reproduction culture($| )' | head -1 || true)"
    if [[ -z "$running" ]]; then
        running="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | head -1 || true)"
    fi
    if [[ -n "$running" ]]; then
        die "the GPU already has work in flight:
  $running
Wait for it, or pass --allow-concurrent if you know the card can take both."
    fi
fi

LOG_DIR="$REPO_ROOT/outputs/culture/logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/base_summary.tsv"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

priority_options=()
[[ ${#first_countries[@]} -gt 0 ]] && priority_options=(--first-countries "${first_countries[@]}")

total=$(( ${#models[@]} * ${#questions[@]} ))
echo "Base models: ${#models[@]} model(s) x ${#questions[@]} question(s) = $total run(s)"
echo "Models:    ${models[*]}"
echo "Questions: ${questions[*]}"
if [[ ${#first_countries[@]} -gt 0 ]]; then
    echo "First:     ${first_countries[*]} (ordering only)"
else
    echo "First:     no country priority"
fi
[[ ${#passthrough[@]} -gt 0 ]] && echo "Run options: ${passthrough[*]}"
echo "Logs:      ${LOG_DIR#"$REPO_ROOT"/}"
echo

if [[ $dry_run -eq 1 ]]; then
    for question in "${questions[@]}"; do
        for model in "${models[@]}"; do
            echo "would run: --models $model --cultures base --questions $question ${priority_options[*]} ${passthrough[*]}"
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
        index=$((index + 1))
        pair="$model/base/$question"
        outputs="$REPO_ROOT/outputs/culture/$model/base/$question"
        log="$LOG_DIR/$model-base-$question.log"

        if [[ $skip_completed -eq 1 && -f "$outputs/capacity.csv" ]]; then
            echo "[$index/$total] $pair — already run, skipping (--redo to rerun)"
            printf '%s\tbase\t%s\tskipped\t0\t\t\t\t\n' "$model" "$question" >> "$SUMMARY"
            skipped=$((skipped + 1))
            continue
        fi

        echo "[$index/$total] $pair — running (log: ${log#"$REPO_ROOT"/})"
        pair_started=$(date +%s)
        if "$REPO_ROOT/scripts/07_culture.sh" \
                --models "$model" \
                --cultures base \
                --questions "$question" \
                --skip-compare \
                "${priority_options[@]}" \
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

        printf '%s\tbase\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$model" "$question" "$status" "$elapsed" "$mass" "$valid" "$prompts" "$rate" >> "$SUMMARY"
        echo "    $status in ${elapsed}s${mass:+, valid-answer mass $mass}${rate:+, answered ${rate}%}"
        if [[ "$status" == "FAILED" ]]; then
            echo "    last lines of $log:"
            tail -5 "$log" | sed 's/^/      /'
        fi
    done
done

echo
if [[ $run_compare -eq 1 && $completed -gt 0 ]]; then
    echo "Rebuilding cross-culture comparison and summary..."
    "$REPO_ROOT/scripts/09_culture_compare.sh" \
        --models "${models[@]}" --questions "${questions[@]}" || true
    "$REPO_ROOT/scripts/14_culture_summary.sh" \
        --models "${models[@]}" --questions "${questions[@]}" || true
    echo
fi

echo "Base run started $started_at, finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  completed: $completed   skipped: $skipped   failed: $failed"
echo "  summary:   ${SUMMARY#"$REPO_ROOT"/}"
column -t -s $'\t' "$SUMMARY" | sed 's/^/  /'

[[ $failed -eq 0 ]]
