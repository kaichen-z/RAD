#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/fetch_upstream.sh METHOD [DESTINATION]

METHOD is one of: anomalib, adaclip, vcpclip, splatpose
DESTINATION defaults to third_party/<upstream-name>.
The checkout is pinned and RAD's small integration adapter is installed when needed.

PIAD is intentionally unavailable here: its public project-page repository does not contain
method source as of this release. See methods/reconstruction/README.md.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="$1"
INSTALLER=""
case "$METHOD" in
  anomalib)
    URL="https://github.com/open-edge-platform/anomalib.git"
    REVISION="2babe0e7902541d75f4f2a71b8769151f5e8a47c"
    NAME="anomalib"
    ;;
  adaclip)
    URL="https://github.com/caoyunkang/AdaCLIP.git"
    REVISION="b762ac40c3f33c77e7e513e48cb436f059d456da"
    NAME="AdaCLIP"
    INSTALLER="$ROOT_DIR/methods/adaclip/install_upstream.py"
    ;;
  vcpclip)
    URL="https://github.com/xiaozhen228/VCP-CLIP.git"
    REVISION="f87a3f6dafe1641fd6b88130393032af3ac0cf65"
    NAME="VCP-CLIP"
    ;;
  splatpose)
    URL="https://github.com/m-kruse98/SplatPose.git"
    REVISION="2c07b6dad268c6ee5c98549ba323acaaa0574916"
    NAME="SplatPose"
    INSTALLER="$ROOT_DIR/methods/reconstruction/install_splatpose.py"
    ;;
  *)
    echo "Unknown or unavailable method: $METHOD" >&2
    usage >&2
    exit 2
    ;;
esac

DESTINATION="${2:-$ROOT_DIR/third_party/$NAME}"
if [[ -e "$DESTINATION" ]]; then
  echo "Destination already exists: $DESTINATION" >&2
  exit 1
fi

git clone --filter=blob:none "$URL" "$DESTINATION"
git -C "$DESTINATION" checkout --detach "$REVISION"
if [[ "$METHOD" == "splatpose" ]]; then
  git -C "$DESTINATION" submodule update --init --recursive
fi
if [[ -n "$INSTALLER" ]]; then
  python "$INSTALLER" "$DESTINATION"
fi
echo "Prepared $METHOD at $DESTINATION ($REVISION)"
