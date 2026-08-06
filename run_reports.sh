#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

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

Everything reads the archived upstream outputs, so no model is loaded and no
inference runs. Tables land in outputs/reports as CSV and as \input-able
booktabs LaTeX under outputs/reports/tex; figures land in figures/reports as
PNG and PDF, drawn without titles or captions for paper.tex.

The robustness section is the slow one: it reads the per-prompt output files
under upstream/extracted/.../data/Robustness and fits a random forest and a
bootstrapped multinomial model.
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
            ./scripts/12_reports.sh "$command_name" --questions all "$@"
        else
            ./scripts/12_reports.sh "$command_name" "$@"
        fi
        ;;
    *) echo "Unknown command: $command_name" >&2; usage >&2; exit 2 ;;
esac
