from pathlib import Path

from rad_benchmark.constants import INSTANCE_TO_CATEGORY
from rad_benchmark.dataset import validate_dataset, write_metadata


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_validate_complete_synthetic_layout(tmp_path: Path) -> None:
    for instance in INSTANCE_TO_CATEGORY:
        _touch(tmp_path / instance / "train" / "good" / "000.png")
        _touch(tmp_path / instance / "test" / "good" / "001.png")
        _touch(tmp_path / instance / "test" / "scratched" / "002.png")
        _touch(tmp_path / instance / "ground_truth" / "scratched" / "002_mask.png")
    report = validate_dataset(tmp_path)
    assert report["valid"]
    assert report["totals"] == {"instances": 18, "images": 54, "masks": 18}


def test_metadata_paths_are_relative(tmp_path: Path) -> None:
    instance = next(iter(INSTANCE_TO_CATEGORY))
    _touch(tmp_path / instance / "test" / "good" / "001.png")
    output = tmp_path / "meta.json"
    metadata = write_metadata(tmp_path, output)
    record = metadata["test"][instance][0]
    assert record["image"] == f"{instance}/test/good/001.png"
    assert record["mask"] is None
