#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

usage() {
    cat <<'EOF'
run_experiment_base_models.sh — The un-finetuned baseline for the culture
experiment.

Usage:
  ./run_experiment_base_models.sh run [--models all|KEY...]
                                      [--questions all|Q...] [--replicates N]
                                      [--first-countries NAME...] [--no-priority]
                                      [--redo] [--no-compare] [--dry-run]
                                      [--allow-concurrent]
  ./run_experiment_base_models.sh compare [--models KEY...] [--questions Q...]
  ./run_experiment_base_models.sh summary [--models KEY...] [--questions Q...]

Models:    gemma4_31b, gemma4_e4b, qwen3_vl_8b, qwen3_vl_2b, llama3_2_3b,
           muse_glimmer_30b, luna, terra, sol
           (default gemma4_31b; --models all for every one; muse_glimmer_30b
           runs alone under the muse extra, selected automatically; luna, terra
           and sol are served through the OpenAI API, stay out of --models all,
           run apart from the local models under the api extra, and need .env)
Questions: d_happy d_polpos d_religiousp d_trust (default: all four)

The base arm is the same weights the culture-finetuned LLMs were fitted from,
sent the paper's own prompts through the same code — the only reference that
isolates what the finetuning did, where Mixtral confounds it with a different
base model.

Results land under outputs/culture/<model>/base/<question>/, beside the
culture-finetuned runs, so `compare` and `summary` pick the base up as another
arm and add base_deltas.csv (each arm minus its own base, same subpopulations)
and fig_home_advantage_base.

Germany's prompts are generated first by default — ordering, not selection; a
finished run is identical either way (--no-priority for the plain order,
--first-countries to name others). Runs are resumable, and the driver refuses
to start while another culture run holds the card.

  ./run_experiment_base_models.sh run --dry-run          # what would run
  ./run_experiment_base_models.sh run --models all       # every base model
  ./run_experiment_base_models.sh run --models all --replicates 3   # FA resampled twice more
  ./run_experiment_base_models.sh summary                # rebuild the tables
EOF
}

command_name="${1:-run}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    -h|--help|help) usage ;;
    run) ./scripts/sweep.sh --base "$@" ;;
    compare) run_python culture-compare "$@" ;;
    summary) run_python culture-summary "$@" ;;
    *) echo "Unknown command: $command_name" >&2; usage >&2; exit 2 ;;
esac
