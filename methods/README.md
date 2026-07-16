# Benchmark method adapters

This directory intentionally contains only RAD-owned adapters, dataset conversion code,
configuration, and prompts. It does **not** vendor full third-party repositories or CUDA
submodules. Install each upstream at the exact revision in [`manifest.toml`](manifest.toml),
then use the corresponding adapter here. This keeps provenance, updates, and license terms
visible.

| Paper family | Baselines | RAD-specific code | What changed for RAD |
|---|---|---|---|
| 2D feature | CFlow, EfficientAD, FastFlow, PaDiM, PatchCore, Reverse Distillation, STFPM, UFlow, WinCLIP | [`src/rad_benchmark/anomalib.py`](../src/rad_benchmark/anomalib.py) | Maps each physical instance's MVTec-style folders into Anomalib `Folder`, keeps normal test images separate, pairs every defect directory with its mask directory, and skips fitting WinCLIP. |
| Zero-shot CLIP | AdaCLIP | [`adaclip/rad_dataset.py`](adaclip/rad_dataset.py) | Loads RAD path metadata without hard-coded roots and preserves semantic-category plus physical-instance labels. |
| Zero-shot CLIP | VCP-CLIP | [`vcpclip/convert_rad.py`](vcpclip/convert_rad.py) | Converts RAD into VCP-CLIP's unified `good`/`anomaly` layout with collision-free names and binary masks. |
| 3D reconstruction | SplatPose, PIAD | [`reconstruction/prepare_splatpose.py`](reconstruction/prepare_splatpose.py), [`reconstruction/run.py`](reconstruction/run.py) | Rewrites camera-frame paths in a separate prepared tree and constructs an explicit SplatPose command. PIAD stops at portable config generation because its method source is not public. |
| VLM | Qwen2.5-VL, GPT-4o | [`vlm/prompt.json`](vlm/prompt.json), [`src/rad_benchmark/vlm.py`](../src/rad_benchmark/vlm.py) | Enforces a provider-neutral JSON response, converts predicted boxes to binary masks, and never stores credentials. |

## Fetch upstreams

The helper clones to an ignored `third_party/` directory and checks out pinned revisions:

```bash
bash scripts/fetch_upstream.sh anomalib
bash scripts/fetch_upstream.sh adaclip
bash scripts/fetch_upstream.sh vcpclip
bash scripts/fetch_upstream.sh splatpose
```

Review the upstream license before downloading weights or running a method. In particular,
SplatPose's Gaussian Splatting components are not covered by RAD's MIT license and may restrict
use or redistribution.

## Smoke checks

These commands do not load a model or dataset:

```bash
python -m rad_benchmark.cli --help
python methods/vcpclip/convert_rad.py --help
python methods/reconstruction/prepare_splatpose.py --help
python methods/reconstruction/run.py --help
bash scripts/fetch_upstream.sh --help
bash scripts/run_clip.sh --help
```

Full experiments require method-specific CUDA environments and checkpoints. PIAD source is not
public as of this release, so its adapter stops after generating a portable config. The paper uses
COLMAP estimates from unmasked images for SplatPose and PIAD; robot ground-truth poses are
metadata, not the reported 3D-baseline input.
