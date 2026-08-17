#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"

usage() {
    cat <<'EOF'
run_reports.sh — reproduce the tables and figures published with
"Machine Bias: How Do Generative Language Models Answer Opinion Polls?".

Usage:
  ./run_reports.sh                                  # every table and figure
  ./run_reports.sh all
  ./run_reports.sh main                             # Tables 2-6, Figures 2,3,4,6
  ./run_reports.sh appendix                         # Tables S1,S2,S4 + Figures S1,S2,S4,S9,S10,S15
  ./run_reports.sh robustness                       # Tables S6,S7,S8,S9 + Figures S12,S13,S14,S16,S17,S18
  ./run_reports.sh tables    [--questions Q...]
  ./run_reports.sh figures   [--series S...]
  ./run_reports.sh smoke                            # one question, two series

Questions: d_happy d_polpos d_religiousp d_trust  (or 'all'; default all here)
Series:    NTP-GPT-4T NTP-Llama-3-70B NTP-Mixtral-8x7B
           FA-GPT-3   FA-Llama-3-70B   FA-Mixtral-8x7B

Everything reads the archived upstream outputs; no model is loaded. Tables
land in outputs/reports as CSV and \input-able booktabs under
outputs/reports/tex; figures in figures/reports as PNG and PDF, untitled for
the paper. robustness is the slow section: it reads the per-prompt Robustness
files and fits a random forest and a bootstrapped multinomial model.
EOF
}

command_name="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    -h|--help|help) usage ;;
    all|main|appendix|robustness|tables|figures|smoke)
        if [[ "$*" != *--questions* ]]; then
            run_python reports "$command_name" --questions all "$@"
        else
            run_python reports "$command_name" "$@"
        fi
        ;;
    *) echo "Unknown command: $command_name" >&2; usage >&2; exit 2 ;;
esac
