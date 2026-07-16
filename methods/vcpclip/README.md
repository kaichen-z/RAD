# VCP-CLIP adapter

VCP-CLIP expects a single `anomaly` directory. Convert RAD without modifying the downloaded
archive:

```bash
python methods/vcpclip/convert_rad.py \
  /data/RAD_with_mask work/vcpclip-rad --link
```

Set the pinned VCP-CLIP checkout's `--data_path` to `work/vcpclip-rad` and use dataset name `rad`.
The pinned implementation routes unknown dataset names through `OtherDataset`; the converter
writes its expected `meta_rad.json`. It preserves the original instance and defect in every
destination filename for auditable pairing.
