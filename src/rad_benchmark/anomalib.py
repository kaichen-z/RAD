"""RAD-specific direct adapter for the Anomalib baselines in the paper."""

from __future__ import annotations

from pathlib import Path

from .constants import INSTANCE_TO_CATEGORY

MODEL_NAMES = (
    "cflow",
    "efficientad",
    "fastflow",
    "padim",
    "patchcore",
    "reversedistillation",
    "stfpm",
    "uflow",
    "winclip",
)


def _imports():
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import (
            Cflow,
            EfficientAd,
            Fastflow,
            Padim,
            Patchcore,
            ReverseDistillation,
            Stfpm,
            Uflow,
            WinClip,
        )
    except ImportError as error:
        raise RuntimeError("Install the Anomalib extra: pip install -e '.[anomalib]'") from error
    models = {
        "cflow": Cflow,
        "efficientad": EfficientAd,
        "fastflow": Fastflow,
        "padim": Padim,
        "patchcore": Patchcore,
        "reversedistillation": ReverseDistillation,
        "stfpm": Stfpm,
        "uflow": Uflow,
        "winclip": WinClip,
    }
    return Folder, Engine, models


def make_datamodule(instance_root: str | Path, batch_size: int, workers: int):
    """Map one RAD instance to Anomalib's supported Folder datamodule."""
    Folder, _, _ = _imports()
    instance_root = Path(instance_root).resolve()
    defect_names = sorted(
        path.name
        for path in (instance_root / "test").iterdir()
        if path.is_dir() and path.name != "good"
    )
    if not defect_names:
        raise ValueError(f"No anomaly directories found under {instance_root / 'test'}")
    return Folder(
        name=f"RAD-{instance_root.name}",
        root=instance_root,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=[f"test/{name}" for name in defect_names],
        mask_dir=[f"ground_truth/{name}" for name in defect_names],
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=workers,
        val_split_mode="none",
    )


def run(
    dataset_root: str | Path,
    output_root: str | Path,
    model_name: str,
    instances: list[str],
    batch_size: int = 8,
    workers: int = 4,
    max_epochs: int = 100,
) -> None:
    """Fit/test one paper baseline over selected physical instances."""
    _, Engine, models = _imports()
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name!r}; choose from {', '.join(MODEL_NAMES)}")
    unknown = sorted(set(instances) - INSTANCE_TO_CATEGORY.keys())
    if unknown:
        raise ValueError(f"Unknown RAD instances: {', '.join(unknown)}")
    dataset_root = Path(dataset_root).resolve()
    for instance in instances:
        datamodule = make_datamodule(dataset_root / instance, batch_size, workers)
        model = models[model_name]()
        engine = Engine(
            default_root_dir=Path(output_root) / model_name / instance,
            accelerator="auto",
            devices=1,
            max_epochs=max_epochs,
        )
        if model_name != "winclip":
            engine.fit(model=model, datamodule=datamodule)
        engine.test(model=model, datamodule=datamodule)
