# RAD public-release statistics

Counts below are derived from the distributed archives, not from mask-file totals. Both RGB variants have identical sample paths and include anomaly masks.

Dataset revision: `5eed55ab545461eb3173ab79e1a468cc0d0f8704`. The published SHA-256 of `Anomaly_refine_msk.zip` is `ba00938286fd511c1c7cec78586435015fe637aa8de3198be583ceae2155fcee`.

## Category counts

| Category | Instances | Train normal | Test normal | Missing | Stained | Scratched | Squeezed | RGB total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Binderclip | 2 | 118 | 18 | 68 | 68 | 68 | 0 | 340 |
| Bowl | 1 | 59 | 9 | 68 | 68 | 60 | 68 | 332 |
| Box | 1 | 59 | 9 | 45 | 68 | 68 | 0 | 249 |
| Can | 1 | 59 | 9 | 41 | 68 | 4 | 68 | 249 |
| Charger | 1 | 59 | 9 | 66 | 66 | 5 | 0 | 205 |
| Cup1 | 1 | 59 | 9 | 68 | 66 | 61 | 68 | 331 |
| Cup2 | 3 | 177 | 27 | 68 | 29 | 40 | 68 | 409 |
| Gluebottle | 2 | 118 | 18 | 53 | 21 | 24 | 68 | 302 |
| Phonecase | 2 | 118 | 18 | 68 | 68 | 68 | 68 | 408 |
| Rubberduck | 1 | 59 | 9 | 68 | 60 | 58 | 68 | 322 |
| Spoon | 1 | 59 | 9 | 68 | 66 | 51 | 68 | 321 |
| Spraybottle | 1 | 59 | 9 | 52 | 62 | 58 | 68 | 308 |
| Tennisball | 1 | 59 | 9 | 67 | 68 | 68 | 0 | 271 |
| **Total** | **18** | **1,062** | **162** | **800** | **778** | **633** | **612** | **4,047** |

There are 1,224 normal RGB images, 2,823 anomalous RGB images, and 2,823 separate ground-truth masks. The test set contains 2,985 RGB images.

## Instance counts

| Instance directory | Train normal | Test normal | Missing | Stained | Scratched | Squeezed | RGB total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `binderclip` | 59 | 9 | 68 | 0 | 68 | 0 | 204 |
| `binderclip2` | 59 | 9 | 0 | 68 | 0 | 0 | 136 |
| `bowl_upright` | 59 | 9 | 68 | 68 | 60 | 68 | 332 |
| `box` | 59 | 9 | 45 | 68 | 68 | 0 | 249 |
| `can` | 59 | 9 | 41 | 68 | 4 | 68 | 249 |
| `charger` | 59 | 9 | 66 | 66 | 5 | 0 | 205 |
| `cup1_upright` | 59 | 9 | 68 | 66 | 61 | 68 | 331 |
| `cup2_upright` | 59 | 9 | 68 | 0 | 0 | 68 | 204 |
| `cup2_upright2` | 59 | 9 | 0 | 29 | 0 | 0 | 97 |
| `cup2_upright3` | 59 | 9 | 0 | 0 | 40 | 0 | 108 |
| `gluebottle` | 59 | 9 | 53 | 0 | 0 | 68 | 189 |
| `gluebottle2` | 59 | 9 | 0 | 21 | 24 | 0 | 113 |
| `phonecase` | 59 | 9 | 68 | 0 | 68 | 68 | 272 |
| `phonecase2` | 59 | 9 | 0 | 68 | 0 | 0 | 136 |
| `rubberduck` | 59 | 9 | 68 | 60 | 58 | 68 | 322 |
| `spoon_upright` | 59 | 9 | 68 | 66 | 51 | 68 | 321 |
| `spraybottle2` | 59 | 9 | 52 | 62 | 58 | 68 | 308 |
| `tennisball` | 59 | 9 | 67 | 68 | 68 | 0 | 271 |

## Reproducibility

The [sample manifest](dataset_manifest.json) lists all 4,047 RGB paths and their corresponding anomaly masks. [Machine-readable statistics](dataset_statistics.json) include category counts, instance counts, and pixel-coverage measurements.

```bash
pip install -e '.[statistics]'
python scripts/dataset_statistics.py Anomaly_refine_msk.zip --output docs
python scripts/plot_dataset_statistics.py docs/dataset_statistics.json \
  --examples assets/defect-examples.png --output assets
```

The plotting script additionally requires matplotlib. Pixel coverage is the mean of `100 * count(mask > 0) / image_area` over anomalous images in each semantic category and defect type; all instances are pooled. Groups without samples are null in the JSON and omitted from the logarithmic plot. This denominator is full image area, not foreground-object area.

## Statistics correction (2026-10-16)

Earlier documentation reported inconsistent totals of 4,287 and 5,848. Neither matches the current distributed archives. The category table overcounted Binderclip by 204, Spraybottle by 10, and Cup2 by 26. Rubberduck missing/scratched counts and Gluebottle missing/stained counts also required correction without changing their category totals. The release contains 68 squeezed Phonecase samples and no squeezed Tennisball samples.

The figures and descriptive statistics have been regenerated from archive entries and all 2,823 anomaly masks. This is a documentation/statistics correction to the existing release. Published benchmark AUROC values are retained as reported; this audit did not rerun the methods.
