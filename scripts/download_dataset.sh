#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/download_dataset.sh {with-masks|without-masks} OUTPUT_FILE

Downloads the hosted RAD archive using gdown. The script does not guess the archive format,
extract files, or overwrite an existing destination.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi
case "$1" in
  with-masks) URL="https://drive.google.com/file/d/1p3v2FeNlHinXFCTZzZujhYhCGUiOL38G/view" ;;
  without-masks) URL="https://drive.google.com/file/d/1cPeHh69vErvC3yZSejyxU4SNK9fVlJWL/view" ;;
  *) echo "Unknown variant: $1" >&2; usage >&2; exit 2 ;;
esac
OUTPUT="$2"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite: $OUTPUT" >&2
  exit 1
fi
if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is required: python -m pip install gdown" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT")"
gdown --fuzzy "$URL" --output "$OUTPUT"
echo "Downloaded to $OUTPUT; extract it, then run rad-benchmark validate."
