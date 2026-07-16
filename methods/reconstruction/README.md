# Reconstruction adapters

The paper evaluates SplatPose and PIAD using camera poses estimated by COLMAP on unmasked
multi-view images. First prepare SplatPose's training tree without changing the source data:

```bash
python methods/reconstruction/prepare_splatpose.py \
  /data/RAD_without_mask work/RAD_3dgs --link
```

Then inspect or launch the pinned upstream command:

```bash
python methods/reconstruction/run.py splatpose \
  --upstream third_party/SplatPose --dataset work/RAD_3dgs \
  --evaluation-dataset /data/RAD_with_mask --instance binderclip --dry-run

python methods/reconstruction/run.py piad-config \
  --dataset /data/RAD_with_mask --instance binderclip
```

Remove `--dry-run` from SplatPose only after following its CUDA, COLMAP, EfficientLoFTR, and
checkpoint setup. Outputs are written outside the upstream checkout. Upstream Gaussian
Splatting license terms apply.

The paper also reports PIAD. Its public project-page repository does not contain the method source
as of this release, so RAD ships only the root-independent dataset/config adapter above and does
not claim an executable reproduction. We will add a pinned launch path if the authors release the
implementation under redistributable terms.
