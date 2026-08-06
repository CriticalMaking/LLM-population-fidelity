#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_URL="https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/89d949782453e711318567af765d39e77c57afb0/mixtral-8x7b-v0.1.Q4_K_M.gguf"
MODEL_SHA256="5e066c60d89d904db46c3bf577661040578a223a5a39ba3fd23ff091549767f0"
DESTINATION="$REPO_ROOT/models/mixtral-8x7b-v0.1.Q4_K_M.gguf"

usage() {
    cat <<'EOF'
download_model.sh — Download and verify the paper's pinned Mixtral GGUF.

Usage:
  ./scripts/download_model.sh [DESTINATION]

Downloads the 26.4 GB Q4_K_M file with resume support and refuses to accept it
unless its SHA-256 matches the provenance manifest.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) DESTINATION="$1" ;;
esac

mkdir -p "$(dirname "$DESTINATION")"
if [[ -f "$DESTINATION" ]] && echo "$MODEL_SHA256  $DESTINATION" | sha256sum --check - >/dev/null 2>&1; then
    echo "Already verified: $DESTINATION"
    exit 0
fi
curl --fail --location --continue-at - --output "$DESTINATION" "$MODEL_URL"
echo "$MODEL_SHA256  $DESTINATION" | sha256sum --check -
echo "Verified model: $DESTINATION"
