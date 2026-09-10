#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
sweep.sh — Drive culture and base-model runs one at a time, with a log and a
summary row per run.

Usage:
  ./scripts/sweep.sh        [--models all|KEY...] [--cultures all|NAME...]
                            [--questions all|Q...] [--first-countries NAME...]
                            [--replicates N] [--redo] [--no-compare] [--dry-run]
                            [run options...]
  ./scripts/sweep.sh --base [--models all|KEY...] [--questions all|Q...]
                            [--first-countries NAME...] [--no-priority]
                            [--replicates N] [--redo] [--no-compare] [--dry-run]
                            [--allow-concurrent] [run options...]

One invocation per model, culture and question, so a failed or interrupted run
never takes the rest of the sweep with it: one log per run, finished runs
skipped unless --redo, cells without a staged adapter skipped and recorded
(german, spanish and spanish-mx are staged for every model), and the
comparison and the MDS plates rebuilt at the end over every model rather
than the ones swept, so a one-model sweep lands in the cross-model plates
instead of replacing them with itself.

The summary TSV is a ledger: this sweep's rows are printed and every cell it
did not run is carried over from the previous sweep, since sweep_cost.csv reads
each cell's status back from it.

--replicates N runs every cell N times over. The first replicate is the run as
it stands and is skipped once complete; each further one resamples the full
answers alone, under its own seed stream, into
outputs/culture/<model>/<arm>/rep<k>/<question>/, next-token probabilities
being deterministic. Replicates are the outermost loop, so the second pass over
every cell finishes before the third begins.

Without --base: every model, german, d_happy (--cultures all for the full
grid). With --base: gemma4_31b on all four questions, cultures frozen to the
base arm, Germany's prompts first (ordering, not selection; --no-priority for
the plain order), the cross-question summary rebuilt too, and a refusal to
start while other work holds the GPU unless --allow-concurrent.

luna, terra and sol are served through the OpenAI API and never join the default
set: name them with --models, on the base variant only, and expect API charges.
They need OPENAI_API_KEY and one <MODEL>_MODEL_ID per model in .env, and they
leave the GPU alone. Probe them with `served-smoke` before paying for a run.
An API model answers in FA only — the served endpoint gives no next-token
probabilities, so its NTP probe is empty and it appears in the FA plates alone.

Examples:
  ./scripts/sweep.sh                                   # every model, german, d_happy
  ./scripts/sweep.sh --models gemma4_31b --cultures all
  ./scripts/sweep.sh --base --models all --dry-run
  ./scripts/sweep.sh --cultures german spanish-mx --questions all --replicates 3

Unrecognised options pass through to the underlying run: --mode, --batch-size,
--fa-max-tokens, --force and --legacy-unseeded-fa all work here.
EOF
}

ALL_MODELS=(gemma4_31b gemma4_e4b qwen3_vl_8b qwen3_vl_2b llama3_2_3b muse_glimmer_30b)
API_MODELS=(luna terra sol)
ALL_CULTURES=(arabic bengali chinese english german korean portuguese spanish spanish-mx turkish)
ALL_QUESTIONS=(d_happy d_polpos d_religiousp d_trust)

base=0
models=()
cultures=()
questions=()
first_countries=()
passthrough=()
prioritize=1
skip_completed=1
run_compare=1
dry_run=0
allow_concurrent=0
replicates=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --base) base=1; shift ;;
        --models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do models+=("$1"); shift; done
            [[ ${#models[@]} -gt 0 ]] || die "--models requires at least one value"
            ;;
        --cultures)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do cultures+=("$1"); shift; done
            [[ ${#cultures[@]} -gt 0 ]] || die "--cultures requires at least one value"
            ;;
        --questions)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do questions+=("$1"); shift; done
            [[ ${#questions[@]} -gt 0 ]] || die "--questions requires at least one value"
            ;;
        --first-countries)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do first_countries+=("$1"); shift; done
            [[ ${#first_countries[@]} -gt 0 ]] || die "--first-countries requires at least one value"
            ;;
        --replicates)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] ||
                die "--replicates requires a whole number of 1 or more"
            replicates="$2"; shift 2
            ;;
        --no-priority) prioritize=0; shift ;;
        --allow-concurrent) allow_concurrent=1; shift ;;
        --redo) skip_completed=0; shift ;;
        --no-compare) run_compare=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

if [[ $base -eq 1 ]]; then
    [[ ${#cultures[@]} -eq 0 ]] || die "--cultures cannot combine with --base: base is the arm"
    cultures=(base)
    [[ ${#models[@]} -eq 0 ]] && models=(gemma4_31b)
    [[ ${#questions[@]} -eq 0 ]] && questions=(all)
else
    [[ $prioritize -eq 1 ]] || die "--no-priority requires --base"
    [[ $allow_concurrent -eq 0 ]] || die "--allow-concurrent requires --base"
    [[ ${#models[@]} -eq 0 ]] && models=(all)
    [[ ${#cultures[@]} -eq 0 ]] && cultures=(german)
    [[ ${#questions[@]} -eq 0 ]] && questions=(d_happy)
fi
[[ "${models[0]}" == "all" ]] && models=("${ALL_MODELS[@]}")
[[ "${cultures[0]}" == "all" ]] && cultures=("${ALL_CULTURES[@]}")
[[ "${questions[0]}" == "all" ]] && questions=("${ALL_QUESTIONS[@]}")

if [[ $prioritize -eq 0 ]]; then
    first_countries=()
elif [[ $base -eq 1 && ${#first_countries[@]} -eq 0 ]]; then
    first_countries=(Germany)
fi

for model in "${models[@]}"; do
    printf '%s\n' "${ALL_MODELS[@]}" "${API_MODELS[@]}" | grep -qx -- "$model" ||
        die "unknown model: $model (known: ${ALL_MODELS[*]} ${API_MODELS[*]})"
done
if [[ $base -eq 0 ]]; then
    for culture in "${cultures[@]}"; do
        printf '%s\n' "${ALL_CULTURES[@]}" | grep -qx -- "$culture" ||
            die "unknown culture: $culture (known: ${ALL_CULTURES[*]})"
    done
fi
for question in "${questions[@]}"; do
    printf '%s\n' "${ALL_QUESTIONS[@]}" | grep -qx -- "$question" ||
        die "unknown question: $question (known: ${ALL_QUESTIONS[*]})"
done

needs_gpu=1
if [[ ${#models[@]} -gt 0 ]]; then
    needs_gpu=0
    for model in "${models[@]}"; do
        printf '%s\n' "${API_MODELS[@]}" | grep -qx -- "$model" || needs_gpu=1
    done
fi

if [[ $base -eq 1 && $dry_run -eq 0 && $allow_concurrent -eq 0 && $needs_gpu -eq 1 ]]; then
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

log_dir="$REPO_ROOT/outputs/culture/logs"
mkdir -p "$log_dir"
if [[ $base -eq 1 ]]; then
    sweep_name="Base sweep"
    summary="$log_dir/base_summary.tsv"
else
    sweep_name="Culture sweep"
    summary="$log_dir/sweep_summary.tsv"
fi
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

priority_options=()
[[ ${#first_countries[@]} -gt 0 ]] && priority_options=(--first-countries "${first_countries[@]}")

set_replicate_arguments() {
    replicate_arguments=()
    [[ "$1" -gt 1 ]] && replicate_arguments=(--replicate "$1" --mode fa)
    true
}

replicate_directory() {
    [[ "$1" -gt 1 ]] && printf '/rep%s' "$1"
    true
}

total=$(( ${#models[@]} * ${#cultures[@]} * ${#questions[@]} * replicates ))
if [[ $base -eq 1 ]]; then
    echo "$sweep_name: ${#models[@]} model(s) x ${#questions[@]} question(s) = $total run(s)"
else
    echo "$sweep_name: ${#models[@]} model(s) x ${#cultures[@]} culture(s) x ${#questions[@]} question(s) = $total run(s)"
fi
echo "Models:    ${models[*]}"
[[ $base -eq 0 ]] && echo "Cultures:  ${cultures[*]}"
echo "Questions: ${questions[*]}"
if [[ ${#first_countries[@]} -gt 0 ]]; then
    echo "First:     ${first_countries[*]} (ordering only)"
elif [[ $base -eq 1 ]]; then
    echo "First:     no country priority"
fi
[[ $replicates -gt 1 ]] && echo "Replicates: $replicates (the first is skipped where complete; the rest resample FA alone)"
[[ ${#passthrough[@]} -gt 0 ]] && echo "Run options: ${passthrough[*]}"
echo "Logs:      ${log_dir#"$REPO_ROOT"/}"
echo

if [[ $dry_run -eq 1 ]]; then
  for replicate in $(seq 1 "$replicates"); do
    set_replicate_arguments "$replicate"
    for question in "${questions[@]}"; do
        for model in "${models[@]}"; do
            for culture in "${cultures[@]}"; do
                if [[ $base -eq 0 && ! -d "$REPO_ROOT/models/culture/$model/$culture" ]]; then
                    echo "would skip: --models $model --cultures $culture (no adapter staged)"
                    continue
                fi
                arguments=(--models "$model" --cultures "$culture" --questions "$question"
                           "${priority_options[@]}" "${passthrough[@]}" "${replicate_arguments[@]}")
                run_outputs="$REPO_ROOT/outputs/culture/$model/$culture$(replicate_directory "$replicate")/$question"
                if [[ $skip_completed -eq 1 && -f "$run_outputs/capacity.csv" ]]; then
                    echo "would skip: ${arguments[*]} (already run)"
                    continue
                fi
                echo "would run: ${arguments[*]}"
            done
        done
    done
  done
    exit 0
fi

require_uv
carried="$(mktemp)"
swept="$(mktemp)"
trap 'rm -f "$carried" "$swept"' EXIT
[[ -f "$summary" ]] && cp "$summary" "$carried"
printf 'model\tculture\tquestion\treplicate\tstatus\tseconds\tvalid_answer_mass\tvalid\tprompts\tvalid_rate\n' > "$summary"

index=0
completed=0
skipped=0
failed=0

for replicate in $(seq 1 "$replicates"); do
  set_replicate_arguments "$replicate"
  for question in "${questions[@]}"; do
  for model in "${models[@]}"; do
    for culture in "${cultures[@]}"; do
        index=$((index + 1))
        run_id="$model/$culture$(replicate_directory "$replicate")/$question"
        run_outputs="$REPO_ROOT/outputs/culture/$run_id"
        log="$log_dir/$model-$culture-$question$(replicate_directory "$replicate" | tr / -).log"

        if [[ $base -eq 0 && ! -d "$REPO_ROOT/models/culture/$model/$culture" ]]; then
            echo "[$index/$total] $run_id — no $culture adapter staged for $model, skipping"
            printf '%s\t%s\t%s\t%s\tno-adapter\t0\t\t\t\t\n' "$model" "$culture" "$question" "$replicate" >> "$summary"
            skipped=$((skipped + 1))
            continue
        fi

        if [[ $skip_completed -eq 1 && -f "$run_outputs/capacity.csv" ]]; then
            echo "[$index/$total] $run_id — already run, skipping (--redo to rerun)"
            printf '%s\t%s\t%s\t%s\tskipped\t0\t\t\t\t\n' "$model" "$culture" "$question" "$replicate" >> "$summary"
            skipped=$((skipped + 1))
            continue
        fi

        echo "[$index/$total] $run_id — running (log: ${log#"$REPO_ROOT"/})"
        run_started=$(date +%s)
        if run_culture \
                --models "$model" \
                --cultures "$culture" \
                --questions "$question" \
                --skip-compare \
                "${priority_options[@]}" \
                "${passthrough[@]}" \
                "${replicate_arguments[@]}" > "$log" 2>&1; then
            status="ok"
            completed=$((completed + 1))
        else
            status="FAILED"
            failed=$((failed + 1))
        fi
        elapsed=$(( $(date +%s) - run_started ))
        mass="$(grep -o 'mean valid-answer mass [0-9.]*' "$log" | tail -1 | awk '{print $NF}')"
        capacity_line="$(grep -o 'capacity: [0-9]*/[0-9]* prompts answered in the paper.s format ([0-9.]*%)' "$log" | tail -1)"
        valid="$(printf '%s' "$capacity_line" | sed -n 's|.*capacity: \([0-9]*\)/.*|\1|p')"
        prompts="$(printf '%s' "$capacity_line" | sed -n 's|.*capacity: [0-9]*/\([0-9]*\) .*|\1|p')"
        rate="$(printf '%s' "$capacity_line" | sed -n 's|.*(\([0-9.]*\)%).*|\1|p')"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$model" "$culture" "$question" "$replicate" "$status" "$elapsed" "$mass" "$valid" "$prompts" "$rate" >> "$summary"
        echo "    $status in ${elapsed}s${mass:+, valid-answer mass $mass}${rate:+, answered ${rate}%}"
        if [[ "$status" == "FAILED" ]]; then
            echo "    last lines of $log:"
            tail -5 "$log" | sed 's/^/      /'
        fi
    done
  done
  done
done

cp "$summary" "$swept"
awk -F'\t' -v OFS='\t' '
    NR == FNR { if (FNR > 1) swept[$1 FS $2 FS $3 FS $4] = 1; next }
    FNR > 1 {
        if (NF == 9) { $0 = $1 OFS $2 OFS $3 OFS 1 OFS $4 OFS $5 OFS $6 OFS $7 OFS $8 OFS $9 }
        if (!($1 FS $2 FS $3 FS $4 in swept)) print
    }
' "$swept" "$carried" >> "$summary"

echo
if [[ $run_compare -eq 1 && $completed -gt 0 ]]; then
    if [[ $base -eq 1 ]]; then
        echo "Rebuilding cross-culture comparison, summary and MDS plates over every model..."
    else
        echo "Rebuilding cross-culture comparison and MDS plates over every model..."
    fi
    run_python culture-compare --questions "${questions[@]}" || true
    if [[ $base -eq 1 ]]; then
        run_python culture-summary --questions "${questions[@]}" || true
    fi
    run_python culture-mds --questions "${questions[@]}" --all-countries --outcomes || true
    echo
fi

echo "$sweep_name started $started_at, finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  completed: $completed   skipped: $skipped   failed: $failed"
echo "  summary:   ${summary#"$REPO_ROOT"/}"
column -t -s $'\t' "$swept" | sed 's/^/  /'

[[ $failed -eq 0 ]]
