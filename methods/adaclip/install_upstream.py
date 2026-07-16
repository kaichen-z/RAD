#!/usr/bin/env python3
"""Install the small RAD dataset registration into a pinned AdaCLIP checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def install(checkout: Path) -> None:
    checkout = checkout.resolve()
    registry = checkout / "dataset" / "__init__.py"
    destination = checkout / "dataset" / "rad.py"
    if not registry.is_file():
        raise FileNotFoundError(f"not an AdaCLIP checkout: {checkout}")
    shutil.copy2(Path(__file__).with_name("upstream_rad.py"), destination)
    text = registry.read_text(encoding="utf-8")
    import_line = "from .rad import RAD_CLS_NAMES, RADDataset, RAD_ROOT\n"
    if import_line not in text:
        marker = "from torch.utils.data import ConcatDataset\n"
        if marker not in text:
            raise RuntimeError("AdaCLIP registry layout changed; refusing an unsafe edit")
        text = text.replace(marker, import_line + marker)
    entry = "    'rad': (RAD_CLS_NAMES, RADDataset, RAD_ROOT),\n"
    if entry not in text:
        marker = "    'visa': (VISA_CLS_NAMES, VisaDataset, VISA_ROOT),\n"
        if marker not in text:
            raise RuntimeError("AdaCLIP registry dictionary changed; refusing an unsafe edit")
        text = text.replace(marker, marker + entry)
    registry.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    install(args.checkout)
    print(f"Installed RAD adapter into {args.checkout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
