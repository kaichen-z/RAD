#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_clip.sh METHOD --upstream DIR --dataset DIR --checkpoint FILE [options]

METHOD: adaclip or vcpclip
Options:
  --output DIR       Output directory (default: outputs/<method>)
  --backbone FILE    VCP-CLIP ViT-L/14@336 checkpoint
  --gpu ID           CUDA device index (default: 0)
  --dry-run          Print the command without executing it
  -h, --help         Show this help

The dataset must first be prepared with the method adapter documented under methods/.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi
METHOD="$1"
shift
UPSTREAM=""
DATASET=""
CHECKPOINT=""
BACKBONE=""
OUTPUT=""
GPU="0"
DRY_RUN="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream) UPSTREAM="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --backbone) BACKBONE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$METHOD" != "adaclip" && "$METHOD" != "vcpclip" ]]; then
  echo "METHOD must be adaclip or vcpclip" >&2
  exit 2
fi
if [[ -z "$UPSTREAM" || -z "$DATASET" || -z "$CHECKPOINT" ]]; then
  echo "--upstream, --dataset, and --checkpoint are required" >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${OUTPUT:-$ROOT_DIR/outputs/$METHOD}"

if [[ "$METHOD" == "adaclip" ]]; then
  COMMAND=(python "$UPSTREAM/test.py" --testing_model dataset --testing_data rad
    --ckt_path "$CHECKPOINT" --save_path "$OUTPUT")
else
  if [[ -z "$BACKBONE" ]]; then
    echo "--backbone is required for vcpclip" >&2
    exit 2
  fi
  COMMAND=(python "$UPSTREAM/test.py" --dataset rad --data_path "$DATASET"
    --checkpoint_path "$CHECKPOINT" --save_path "$OUTPUT"
    --pretrained_path "$BACKBONE" --prompt_len 2 --deep_prompt_len 1
    --device_id "$GPU" --features_list 6 12 18 24 --pretrained openai
    --image_size 518 --seed 333
    --config_path "$UPSTREAM/models/model_configs/ViT-L-14-336.json"
    --model ViT-L-14-336)
fi

printf 'Command:'
if [[ "$METHOD" == "adaclip" ]]; then
  printf ' RAD_DATA_ROOT=%q' "$DATASET"
fi
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
mkdir -p "$OUTPUT"
if [[ "$METHOD" == "adaclip" ]]; then
  RAD_DATA_ROOT="$DATASET" CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "${COMMAND[@]}"
else
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "${COMMAND[@]}"
fi
