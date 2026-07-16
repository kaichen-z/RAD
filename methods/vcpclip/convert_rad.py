#!/usr/bin/env python3
"""Convert RAD into VCP-CLIP's unified good/anomaly representation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rad_benchmark.constants import INSTANCE_TO_CATEGORY
from rad_benchmark.dataset import expected_mask, image_files


def transfer(source: Path, destination: Path, link: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    if link:
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def convert(source: Path, destination: Path, link: bool = False) -> list[dict[str, object]]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination must be empty: {destination}")
    records: list[dict[str, object]] = []
    for instance, category in INSTANCE_TO_CATEGORY.items():
        instance_root = source / instance
        if not instance_root.is_dir():
            raise FileNotFoundError(f"missing RAD instance: {instance_root}")
        for split in ("train", "test"):
            split_root = instance_root / split
            for defect_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                abnormal = defect_dir.name != "good"
                target_kind = "anomaly" if abnormal else "good"
                for image in image_files(defect_dir):
                    unique_stem = f"{instance}__{defect_dir.name}__{image.stem}"
                    target = (
                        destination
                        / instance
                        / split
                        / target_kind
                        / f"{unique_stem}{image.suffix.lower()}"
                    )
                    transfer(image, target, link)
                    mask_target: Path | None = None
                    if abnormal:
                        mask = expected_mask(
                            image, instance_root / "ground_truth" / defect_dir.name
                        )
                        if mask is None:
                            raise FileNotFoundError(f"missing mask for {image}")
                        mask_target = (
                            destination
                            / instance
                            / "ground_truth"
                            / "anomaly"
                            / f"{unique_stem}.png"
                        )
                        transfer(mask, mask_target, link)
                    records.append(
                        {
                            "image": target.relative_to(destination).as_posix(),
                            "mask": (
                                mask_target.relative_to(destination).as_posix()
                                if mask_target
                                else None
                            ),
                            "category": category,
                            "instance": instance,
                            "defect": defect_dir.name,
                            "anomaly": int(abnormal),
                        }
                    )
    destination.mkdir(parents=True, exist_ok=True)
    grouped = {"train": {}, "test": {}}
    for record in records:
        split = Path(str(record["image"])).parts[1]
        instance = str(record["instance"])
        grouped[split].setdefault(instance, []).append(
            {
                "img_path": record["image"],
                "mask_path": record["mask"] or "",
                "cls_name": instance,
                "specie_name": record["defect"],
                "anomaly": record["anomaly"],
            }
        )
    (destination / "meta_rad.json").write_text(
        json.dumps(grouped, indent=2) + "\n", encoding="utf-8"
    )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="RAD_with_mask root")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--link", action="store_true", help="symlink images instead of copying")
    args = parser.parse_args(argv)
    records = convert(args.source, args.destination, args.link)
    print(f"Converted {len(records)} samples to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
