"""Direct implementation of the paper's VLM box-to-mask protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _coerce_box(
    box: list[float], width: int, height: int, normalized: bool
) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("bbox must contain [x_min, y_min, x_max, y_max]")
    x1, y1, x2, y2 = (float(value) for value in box)
    if normalized:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    left, right = sorted((max(0, min(width, round(x1))), max(0, min(width, round(x2)))))
    top, bottom = sorted((max(0, min(height, round(y1))), max(0, min(height, round(y2)))))
    return left, top, right, bottom


def render_response_mask(response: dict[str, Any], width: int, height: int):
    """Render a parsed VLM response to a binary Pillow image."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - exercised by installation smoke tests
        raise RuntimeError("Install the base project dependencies to render masks") from error

    mask = Image.new("L", (width, height), color=0)
    if not response.get("is_anomalous", False):
        return mask
    boxes = response.get("boxes", [])
    normalized = response.get("coordinate_space", "pixels") == "normalized"
    draw = ImageDraw.Draw(mask)
    for item in boxes:
        box = item["bbox"] if isinstance(item, dict) else item
        left, top, right, bottom = _coerce_box(box, width, height, normalized)
        if right > left and bottom > top:
            draw.rectangle((left, top, right - 1, bottom - 1), fill=255)
    return mask


def responses_to_masks(
    input_json: str | Path, image_root: str | Path, output_root: str | Path
) -> int:
    """Convert JSON responses to masks while preserving image-relative paths."""
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install the base project dependencies to render masks") from error

    entries = json.loads(Path(input_json).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("VLM response file must contain a JSON list")
    image_root = Path(image_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    for entry in entries:
        relative = Path(entry["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"image path must be relative to image_root: {relative}")
        image_path = image_root / relative
        with Image.open(image_path) as image:
            mask = render_response_mask(entry, *image.size)
        destination = output_root / relative.parent / f"{relative.stem}_mask.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        mask.save(destination)
    return len(entries)
