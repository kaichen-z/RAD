# AdaCLIP adapter

Fetch AdaCLIP (which installs the small RAD registry), generate AdaCLIP-compatible metadata at the
dataset root, and run through the parameterized wrapper:

```bash
bash scripts/fetch_upstream.sh adaclip
rad-benchmark metadata /data/RAD_with_mask /data/RAD_with_mask/meta.json
python methods/adaclip/rad_dataset.py /data/RAD_with_mask/meta.json --summary
bash scripts/run_clip.sh adaclip --upstream third_party/AdaCLIP \
  --dataset /data/RAD_with_mask --checkpoint /weights/adaclip.pth --dry-run
```

`rad_dataset.py` is a direct, root-independent metadata dataset; `upstream_rad.py` is the thin
AdaCLIP `BaseDataset` registration installed by `install_upstream.py`. The wrapper passes the data
root explicitly through `RAD_DATA_ROOT`, which is visible in `--dry-run` output. Use the official
checkpoint instructions.
RAD does not redistribute AdaCLIP weights or tokenizer assets.
