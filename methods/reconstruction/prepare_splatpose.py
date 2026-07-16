#!/usr/bin/env python3
"""Prepare RAD normal views and camera transforms for a SplatPose checkout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rad_benchmark.constants import INSTANCE_TO_CATEGORY


def find_frame_image(instance: Path, file_path: str) -> Path:
    relative = Path(file_path.removeprefix("./"))
    candidates = [instance / relative]
    if relative.suffix:
        candidates.append(instance / "train" / "good" / relative.name)
    else:
        for suffix in (".png", ".jpg", ".jpeg"):
            candidates.extend(
                (
                    instance / f"{relative}{suffix}",
                    instance / "train" / "good" / f"{relative.name}{suffix}",
                )
            )
    match = next((path for path in candidates if path.is_file()), None)
    if match is None:
        raise FileNotFoundError(f"cannot resolve transform frame {file_path!r} under {instance}")
    return match


def transfer(source: Path, destination: Path, link: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    if link:
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def prepare_instance(source: Path, destination: Path, name: str, link: bool) -> int:
    instance = source / name
    transforms_path = instance / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"missing camera metadata: {transforms_path}")
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"no frames in {transforms_path}")
    prepared_frames = []
    for index, frame in enumerate(frames):
        image = find_frame_image(instance, frame["file_path"])
        target = destination / name / "train" / f"train_{index:03d}{image.suffix.lower()}"
        transfer(image, target, link)
        if "transform_matrix" not in frame:
            raise ValueError(f"frame {index} in {transforms_path} has no transform_matrix")
        prepared_frames.append(
            {"file_path": f"./train/{target.name}", "transform_matrix": frame["transform_matrix"]}
        )
    output = {key: value for key, value in transforms.items() if key != "frames"}
    output["frames"] = prepared_frames
    output_path = destination / name / "transforms_train.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return len(prepared_frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="RAD archive with COLMAP-estimated transforms.json"
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument("--instances", nargs="+", choices=tuple(INSTANCE_TO_CATEGORY))
    parser.add_argument(
        "--link", action="store_true", help="symlink normal views instead of copying"
    )
    args = parser.parse_args(argv)
    instances = args.instances or list(INSTANCE_TO_CATEGORY)
    total = 0
    for instance in instances:
        total += prepare_instance(
            args.source.resolve(), args.destination.resolve(), instance, args.link
        )
    print(f"Prepared {total} normal views across {len(instances)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
