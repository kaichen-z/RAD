"""AdaCLIP BaseDataset registration for RAD (installed by install_upstream.py)."""

import os

from .base_dataset import BaseDataset

RAD_CLS_NAMES = [
    "binderclip", "binderclip2", "bowl_upright", "box", "can", "charger",
    "cup1_upright", "cup2_upright", "cup2_upright2", "cup2_upright3",
    "gluebottle", "gluebottle2", "phonecase", "phonecase2", "rubberduck",
    "spoon_upright", "spraybottle2", "tennisball",
]
RAD_ROOT = os.environ.get("RAD_DATA_ROOT", "datasets/RAD_with_mask")


class RADDataset(BaseDataset):
    def __init__(
        self,
        transform,
        target_transform,
        clsnames=RAD_CLS_NAMES,
        aug_rate=0.2,
        root=RAD_ROOT,
        training=True,
    ):
        super().__init__(
            clsnames=clsnames,
            transform=transform,
            target_transform=target_transform,
            root=root,
            aug_rate=aug_rate,
            training=training,
        )
