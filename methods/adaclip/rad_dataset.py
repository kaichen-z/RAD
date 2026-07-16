#!/usr/bin/env python3
"""Root-independent RAD dataset adapter for AdaCLIP-style evaluation.

This file is RAD-owned integration code. AdaCLIP itself remains in its upstream
repository and retains its MIT license and attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class RADDataset:
    """Minimal path dataset backed by `rad-benchmark metadata` output.

    The adapter deliberately returns paths and labels rather than importing a
    particular AdaCLIP transform stack. A pinned upstream checkout can apply its
    own image and target transforms without this repository copying model code.
    """

    def __init__(self, metadata: str | Path, split: str = "test") -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        payload = json.loads(Path(metadata).read_text(encoding="utf-8"))
        if split not in payload or not isinstance(payload[split], dict):
            raise ValueError(f"metadata does not contain a {split!r} mapping")
        self.records: list[dict[str, Any]] = [
            record for instance in payload[split].values() for record in instance
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    dataset = RADDataset(args.metadata, args.split)
    if args.summary:
        instances = sorted({record["instance"] for record in dataset.records})
        anomalous = sum(int(record["anomaly"]) for record in dataset.records)
        print(f"split={args.split} samples={len(dataset)} anomalous={anomalous}")
        print("instances=" + ",".join(instances))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
