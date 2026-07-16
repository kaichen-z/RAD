"""Command-line interface for release-safe RAD utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anomalib import MODEL_NAMES
from .constants import INSTANCE_TO_CATEGORY
from .dataset import validate_dataset, write_metadata
from .vlm import responses_to_masks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rad-benchmark", description="RAD benchmark utilities")
    parser.add_argument("--version", action="version", version="rad-benchmark 2.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a downloaded RAD archive")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--require-poses", action="store_true")
    validate.add_argument("--output", type=Path, help="optional JSON report path")

    metadata = subparsers.add_parser("metadata", help="write CLIP-adapter metadata")
    metadata.add_argument("dataset", type=Path)
    metadata.add_argument("output", type=Path)

    masks = subparsers.add_parser("vlm-masks", help="turn VLM bounding boxes into binary masks")
    masks.add_argument("responses", type=Path)
    masks.add_argument("image_root", type=Path)
    masks.add_argument("output_root", type=Path)

    anomalib = subparsers.add_parser("anomalib", help="run an Anomalib paper baseline")
    anomalib.add_argument("dataset", type=Path)
    anomalib.add_argument("output", type=Path)
    anomalib.add_argument("--model", required=True, choices=MODEL_NAMES)
    anomalib.add_argument(
        "--instances",
        nargs="+",
        choices=tuple(INSTANCE_TO_CATEGORY),
        default=list(INSTANCE_TO_CATEGORY),
    )
    anomalib.add_argument("--batch-size", type=int, default=8)
    anomalib.add_argument("--workers", type=int, default=4)
    anomalib.add_argument("--max-epochs", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        report = validate_dataset(args.dataset, require_poses=args.require_poses)
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0 if report["valid"] else 1
    if args.command == "metadata":
        samples = write_metadata(args.dataset, args.output)
        count = sum(len(records) for split in samples.values() for records in split.values())
        print(f"Wrote {count} records to {args.output}")
        return 0
    if args.command == "vlm-masks":
        count = responses_to_masks(args.responses, args.image_root, args.output_root)
        print(f"Wrote {count} masks under {args.output_root}")
        return 0
    if args.command == "anomalib":
        from .anomalib import run

        run(
            args.dataset,
            args.output,
            args.model,
            args.instances,
            args.batch_size,
            args.workers,
            args.max_epochs,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
