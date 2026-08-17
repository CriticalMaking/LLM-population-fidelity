#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
16_adapter_health.sh — Check staged culture adapters before spending GPU time.

Usage:
  ./scripts/16_adapter_health.sh [--models KEY...] [--cultures NAME...]

A run whose adapter cannot answer the paper's prompt costs about an hour a
question to discover through inference. The same fact is visible in seconds
from the adapter itself, so this reads it there and writes
outputs/culture/adapter_health.csv with one row per staged adapter.

Two independent signals, because they answer different questions:

  eval_token_accuracy   what the finetuning learned, from the trainer's own
                        end-of-training numbers
  update_norm_mean      how large a weight update the adapter applies, which is
                        how a failed run shows up in the weights

  first_loss            cross-entropy at the first logged step, against
  random_guess_loss     ln(vocab_size). A first loss above that ceiling means
                        the base model was already emitting confidently wrong
                        logits before training began — a broken base, not a
                        finetuning that drifted, and a different repository to
                        fix.

Verdicts are healthy, diverged, diverged (broken base), or unknown when no
trainer state was staged beside the weights. Unknown is never treated as a
pass: an adapter nobody measured is not an adapter known to be sound.

The CSV is the inventory of every staged adapter, so --models or --cultures
prints those rows without truncating the file to them; the unfiltered run is
what rewrites it.

Loads no model and reads no GPU, so it is safe to run at any time. `adapters`
prints the same lines after staging, so the usual path never needs this
explicitly.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

run_python culture-health "$@"
