#!/usr/bin/env python3
"""Install RAD's three-line data-root adapter into pinned SplatPose."""

from __future__ import annotations

import argparse
from pathlib import Path


def install(checkout: Path) -> None:
    entrypoint = checkout.resolve() / "train_and_render.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"not a SplatPose checkout: {checkout}")
    text = entrypoint.read_text(encoding="utf-8")
    argument = (
        'pre_parser.add_argument("-rad_data_path", type=str, '
        'help="RAD evaluation archive with masks", required=True)'
    )
    if argument not in text:
        lines = text.splitlines()
        index = next(
            (i for i, line in enumerate(lines) if 'add_argument("-data_path"' in line), None
        )
        if index is None:
            raise RuntimeError("SplatPose argument layout changed; refusing an unsafe edit")
        lines.insert(index + 1, argument)
        text = "\n".join(lines) + "\n"
    config_line = '    "rad_data_dir" : lego_args.rad_data_path,'
    if config_line not in text:
        marker = '    "data_dir" : data_base_dir,'
        if marker not in text:
            raise RuntimeError("SplatPose config layout changed; refusing an unsafe edit")
        text = text.replace(marker, marker + "\n" + config_line, 1)
    if 'data_dir=config["rad_data_dir"])' not in text:
        marker = "data_dir=None)"
        if marker not in text:
            raise RuntimeError("SplatPose evaluation call changed; refusing an unsafe edit")
        text = text.replace(marker, 'data_dir=config["rad_data_dir"])', 1)
    entrypoint.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    install(args.checkout)
    print(f"Installed RAD adapter into {args.checkout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
