#!/usr/bin/env python3
"""Recount RAD ZIP entries and measure full-image anomaly-mask coverage.

Usage: python scripts/dataset_statistics.py Anomaly_refine_msk.zip --output docs
The optional mask archive supports auditing independently range-fetched masks.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rad_benchmark.constants import (  # noqa: E402
    DEFECT_TYPES,
    INSTANCE_TO_CATEGORY,
    SEMANTIC_CATEGORIES,
)

REVISION = "5eed55ab545461eb3173ab79e1a468cc0d0f8704"
ARCHIVE_SHA256 = "ba00938286fd511c1c7cec78586435015fe637aa8de3198be583ceae2155fcee"


def summarize(archive_path: Path, mask_archive_path: Path | None = None) -> tuple[dict, dict]:
    """Count RGB samples once; masks are annotations, not additional RGB images."""
    instances = {name: Counter() for name in INSTANCE_TO_CATEGORY}
    coverage = defaultdict(list)
    samples = []
    with zipfile.ZipFile(archive_path) as archive, zipfile.ZipFile(
        mask_archive_path or archive_path
    ) as masks:
        members = set(archive.namelist())
        for info in sorted(archive.infolist(), key=lambda entry: entry.filename):
            parts = info.filename.split("/")
            if info.is_dir() or len(parts) != 5 or parts[2] not in ("train", "test"):
                continue
            root, instance, split, defect, filename = parts
            if instance not in INSTANCE_TO_CATEGORY or not filename.lower().endswith(".png"):
                raise ValueError(f"Unexpected sample: {info.filename}")
            if split == "train" and defect != "good":
                raise ValueError(f"Unexpected anomalous training sample: {info.filename}")
            category = INSTANCE_TO_CATEGORY[instance]
            kind = f"{split}_normal" if defect == "good" else defect
            instances[instance][kind] += 1
            mask = None
            if defect != "good":
                if defect not in DEFECT_TYPES:
                    raise ValueError(f"Unknown defect: {defect}")
                mask = f"{instance}/ground_truth/{defect}/{Path(filename).stem}_mask.png"
                full_mask = f"{root}/{mask}"
                if full_mask not in members:
                    raise ValueError(f"Missing mask: {full_mask}")
                with Image.open(masks.open(full_mask)) as image:
                    values = np.asarray(image.convert("L"))
                # Zero is background; every positive label is anomalous.
                ratio = float(np.count_nonzero(values) / values.size)
                coverage[(category, defect)].append(ratio)
            samples.append({
                "category": category, "instance": instance, "split": split,
                "defect": defect, "image": "/".join(parts[1:]), "mask": mask,
            })
    columns = ("train_normal", "test_normal", *DEFECT_TYPES)
    categories = {}
    for category in SEMANTIC_CATEGORIES:
        names = [n for n, c in INSTANCE_TO_CATEGORY.items() if c == category]
        counts = Counter()
        for name in names:
            counts.update(instances[name])
        row = {key: counts[key] for key in columns}
        row.update(instances=len(names), normal=counts["train_normal"] + counts["test_normal"],
                   anomalous=sum(counts[d] for d in DEFECT_TYPES), images=sum(counts.values()))
        row["mean_anomalous_pixel_percent"] = {
            defect: (100 * float(np.mean(coverage[(category, defect)]))
                     if coverage[(category, defect)] else None)
            for defect in DEFECT_TYPES
        }
        categories[category] = row
    totals = {key: sum(row[key] for row in categories.values()) for key in columns}
    totals.update(normal=sum(r["normal"] for r in categories.values()),
                  anomalous=sum(r["anomalous"] for r in categories.values()),
                  images=len(samples), masks=sum(r["anomalous"] for r in categories.values()),
                  semantic_categories=len(categories), instances=len(instances))
    source = {
        "dataset": "zhouk777/RAD", "revision": REVISION,
        "archive": "Anomaly_refine_msk.zip", "published_archive_sha256": ARCHIVE_SHA256,
    }
    statistics = {
        "source": source, "totals": totals,
        "pixel_coverage_definition": (
            "For each anomalous image: 100 * count(mask > 0) / number of image pixels. "
            "Report the arithmetic mean over images for each semantic category and defect; "
            "pool all corresponding instance directories. Null means no samples."
        ),
        "categories": categories,
        "instances": {name: {key: row[key] for key in columns}
                      for name, row in instances.items()},
    }
    manifest = {"source": source, "samples": samples}
    return statistics, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--mask-archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    statistics, manifest = summarize(args.archive, args.mask_archive)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, result in [("dataset_statistics.json", statistics),
                             ("dataset_manifest.json", manifest)]:
        (args.output / filename).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(statistics["totals"], indent=2))


if __name__ == "__main__":
    main()
