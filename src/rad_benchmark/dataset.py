"""Validation and metadata generation for RAD's MVTec-style layout."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import IMAGE_SUFFIXES, INSTANCE_TO_CATEGORY


def image_files(directory: Path) -> list[Path]:
    """Return image files in stable order without following unrelated files."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def expected_mask(image: Path, mask_dir: Path) -> Path | None:
    """Resolve the two mask naming conventions found in RAD-style archives."""
    candidates = (
        mask_dir / f"{image.stem}_mask.png",
        mask_dir / f"{image.stem}.png",
    )
    return next((path for path in candidates if path.is_file()), None)


@dataclass
class InstanceReport:
    name: str
    category: str
    train_normal: int = 0
    test_normal: int = 0
    anomalous: dict[str, int] = field(default_factory=dict)
    masks: int = 0
    has_transforms: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def images(self) -> int:
        return self.train_normal + self.test_normal + sum(self.anomalous.values())


def validate_instance(root: Path, name: str, require_poses: bool = False) -> InstanceReport:
    """Validate one physical object instance and count its samples."""
    base = root / name
    report = InstanceReport(name=name, category=INSTANCE_TO_CATEGORY[name])
    report.train_normal = len(image_files(base / "train" / "good"))
    report.test_normal = len(image_files(base / "test" / "good"))
    if report.train_normal == 0:
        report.errors.append("train/good has no images")
    if report.test_normal == 0:
        report.errors.append("test/good has no images")

    test_dir = base / "test"
    defect_dirs = sorted(
        path for path in test_dir.iterdir() if path.is_dir() and path.name != "good"
    ) if test_dir.is_dir() else []
    if not defect_dirs:
        report.errors.append("test has no anomaly directories")

    for defect_dir in defect_dirs:
        images = image_files(defect_dir)
        report.anomalous[defect_dir.name] = len(images)
        mask_dir = base / "ground_truth" / defect_dir.name
        for image in images:
            if expected_mask(image, mask_dir) is None:
                report.errors.append(f"missing mask for test/{defect_dir.name}/{image.name}")
            else:
                report.masks += 1

    pose_candidates = (base / "transforms.json", base / "transforms_train.json")
    report.has_transforms = any(path.is_file() for path in pose_candidates)
    if require_poses and not report.has_transforms:
        report.errors.append("camera-pose metadata is missing")
    return report


def validate_dataset(root: str | Path, require_poses: bool = False) -> dict[str, Any]:
    """Validate all expected RAD instances and return a machine-readable report."""
    root = Path(root).expanduser().resolve()
    reports: list[InstanceReport] = []
    errors: list[str] = []
    if not root.is_dir():
        return {"root": str(root), "valid": False, "errors": ["dataset root does not exist"]}

    for name in INSTANCE_TO_CATEGORY:
        if not (root / name).is_dir():
            errors.append(f"missing instance directory: {name}")
            continue
        reports.append(validate_instance(root, name, require_poses=require_poses))

    unexpected = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name not in INSTANCE_TO_CATEGORY
    )
    errors.extend(
        error for report in reports for error in (f"{report.name}: {e}" for e in report.errors)
    )
    return {
        "root": str(root),
        "valid": not errors,
        "instances": [asdict(report) | {"images": report.images} for report in reports],
        "totals": {
            "instances": len(reports),
            "images": sum(report.images for report in reports),
            "masks": sum(report.masks for report in reports),
        },
        "unexpected_directories": unexpected,
        "errors": errors,
    }


def write_metadata(root: str | Path, output: str | Path) -> dict[str, Any]:
    """Write path-only metadata consumed by CLIP baseline adapters."""
    root = Path(root).expanduser().resolve()
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {"train": {}, "test": {}}
    for instance, category in INSTANCE_TO_CATEGORY.items():
        base = root / instance
        if not base.is_dir():
            continue
        for split in ("train", "test"):
            records: list[dict[str, Any]] = []
            split_dir = base / split
            if split_dir.is_dir():
                for defect_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                    abnormal = defect_dir.name != "good"
                    for image in image_files(defect_dir):
                        mask = expected_mask(image, base / "ground_truth" / defect_dir.name)
                        records.append(
                            {
                                "image": image.relative_to(root).as_posix(),
                                "mask": mask.relative_to(root).as_posix() if mask else None,
                                "img_path": image.relative_to(root).as_posix(),
                                "mask_path": mask.relative_to(root).as_posix() if mask else "",
                                "instance": instance,
                                "category": category,
                                "cls_name": instance,
                                "defect": defect_dir.name,
                                "specie_name": defect_dir.name,
                                "anomaly": int(abnormal),
                            }
                        )
            samples[split][instance] = records
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    return samples
