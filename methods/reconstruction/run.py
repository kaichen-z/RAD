#!/usr/bin/env python3
"""Run SplatPose or generate the release-safe RAD configuration for PIAD."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from rad_benchmark.constants import INSTANCE_TO_CATEGORY


def build_splatpose_command(
    upstream: Path,
    prepared_dataset: Path,
    evaluation_dataset: Path,
    instance: str,
) -> list[str]:
    entrypoint = upstream / "train_and_render.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"missing pinned SplatPose entrypoint: {entrypoint}")
    return [
        sys.executable,
        str(entrypoint),
        "-c",
        instance,
        "-data_path",
        str(prepared_dataset),
        "-rad_data_path",
        str(evaluation_dataset),
        "-p",
        "rad",
    ]


def write_piad_config(dataset: Path, instance: str, output: Path) -> Path:
    """Write path-explicit PIAD configuration without requiring unreleased source."""
    config = output / "configs" / f"{instance}.txt"
    config.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"model_name = {instance}\n"
        f"output_dir = {output / 'output'}\n"
        "dataset_type = blender\n"
        f"data_dir = {dataset}\n"
        f"reflection_dir = {output / 'reflection'}\n"
        f"ckpt_dir = {output / 'checkpoints'}\n"
        f"ckpt_name = {instance}\n"
        "white_background = True\n"
        "lrate = 0.02\n"
    )
    config.write_text(content, encoding="utf-8")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("splatpose", "piad-config"))
    parser.add_argument("--upstream", type=Path, help="pinned SplatPose checkout")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path)
    parser.add_argument("--instance", choices=tuple(INSTANCE_TO_CATEGORY), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction"))
    parser.add_argument("--gpu", help="CUDA device index; omitted leaves environment unchanged")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()

    if args.method == "piad-config":
        config = write_piad_config(args.dataset.resolve(), args.instance, output)
        print(f"Wrote RAD PIAD config to {config}")
        print("PIAD method source is not public as of this release; no launch was attempted.")
        return 0

    if args.upstream is None:
        parser.error("splatpose requires --upstream")
    if args.evaluation_dataset is None:
        parser.error("splatpose requires --evaluation-dataset")
    command = build_splatpose_command(
        args.upstream.resolve(),
        args.dataset.resolve(),
        args.evaluation_dataset.resolve(),
        args.instance,
    )
    print(shlex.join(command))
    if args.dry_run:
        return 0
    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=args.upstream, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
