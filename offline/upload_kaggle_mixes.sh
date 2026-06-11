#!/usr/bin/env bash
# Upload all corpus/mix_*.jsonl files as Kaggle dataset slug "nemotron-mixes".
#
# Usage (from repo root):
#   bash offline/upload_kaggle_mixes.sh [--version "message"]
#
# Prerequisites:
#   kaggle CLI installed and configured (~/.kaggle/kaggle.json with username/key).
#   corpus/mix_*.jsonl files already built by offline/build_corpus.py.
#
# Size note: each mix file is roughly 0.5–0.7 GB; 10 files ≈ 5–7 GB total.
# Kaggle dataset limit is 100 GB, so a single dataset is fine unless individual
# files are very large.  If you need to gzip: gzip corpus/mix_*.jsonl  and update
# CORPUS_PATH in exp files to /kaggle/input/nemotron-mixes/mix_<tag>.jsonl.gz,
# then decompress at notebook start.  Do NOT gzip by default — the trainer reads
# .jsonl directly.
#
# If you split across multiple datasets (e.g. one per exp), set each exp's
# CORPUS_PATH to /kaggle/input/<slug>/mix_<tag>.jsonl and attach that dataset.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS_DIR="$REPO_ROOT/corpus"
DATASET_SLUG="nemotron-mixes"          # Kaggle will prefix your username automatically
VERSION_MSG="${1:-"add mix files"}"

# ── Sanity check ────────────────────────────────────────────────────────
if ! command -v kaggle &>/dev/null; then
  echo "ERROR: kaggle CLI not found. Install with: pip install kaggle" >&2
  exit 1
fi

MIX_COUNT=$(find "$CORPUS_DIR" -maxdepth 1 -name "mix_*.jsonl" | wc -l | tr -d ' ')
if [[ "$MIX_COUNT" -eq 0 ]]; then
  echo "ERROR: no corpus/mix_*.jsonl files found. Run offline/build_corpus.py first." >&2
  exit 1
fi
echo "Found $MIX_COUNT mix file(s) in $CORPUS_DIR"

# ── Write dataset-metadata.json if not present ──────────────────────────
META="$CORPUS_DIR/dataset-metadata.json"
KAGGLE_USERNAME="$(kaggle config view 2>/dev/null | grep username | awk '{print $2}')"
if [[ -z "$KAGGLE_USERNAME" ]]; then
  echo "ERROR: could not read Kaggle username from config." >&2
  exit 1
fi

if [[ ! -f "$META" ]]; then
  cat > "$META" <<EOF
{
  "title": "$DATASET_SLUG",
  "id": "$KAGGLE_USERNAME/$DATASET_SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
  echo "Created $META"
fi

# ── Create or version ────────────────────────────────────────────────────
# Check if the dataset already exists on Kaggle.
if kaggle datasets status "$KAGGLE_USERNAME/$DATASET_SLUG" &>/dev/null 2>&1; then
  echo "Dataset $KAGGLE_USERNAME/$DATASET_SLUG exists — creating new version..."
  kaggle datasets version -p "$CORPUS_DIR" -m "$VERSION_MSG" --dir-mode zip
else
  echo "Dataset $KAGGLE_USERNAME/$DATASET_SLUG not found — creating..."
  kaggle datasets create -p "$CORPUS_DIR"
fi

echo ""
echo "Done. Attach dataset  $KAGGLE_USERNAME/$DATASET_SLUG  to each exp notebook."
echo "Each exp reads from /kaggle/input/$DATASET_SLUG/mix_<tag>.jsonl"
