# Anomalib adapter

Install the project with `pip install -e '.[anomalib]'`, validate the archive, then run one
of the nine paper baselines. The first eight are fitted on normal RAD images; WinCLIP is
zero-shot and is tested directly.

```bash
rad-benchmark validate /data/RAD_with_mask
rad-benchmark anomalib /data/RAD_with_mask outputs/anomalib \
  --model patchcore --instances binderclip
```

Run separate processes for different GPUs. Results and downloaded weights are ignored by
Git. The adapter targets the pinned Anomalib revision in `../manifest.toml` and uses public
Anomalib APIs only.
