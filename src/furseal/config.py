"""Project paths and configuration.

All pipeline stages read/write below a single data root so the code runs
unchanged on a laptop, a lab machine, or Google Colab. The root is resolved
in this order:

1. ``--data-root`` CLI argument (each script exposes it)
2. ``FURSEAL_DATA_ROOT`` environment variable
3. ``./data`` relative to the current working directory
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def default_data_root() -> Path:
    return Path(os.environ.get("FURSEAL_DATA_ROOT", "data"))


@dataclass
class ProjectPaths:
    """Canonical dataset layout used across the pipeline.

    Mirrors the folder structure the original notebooks used on Google
    Drive (``.../Capstone/Deakin/furseal/``), rooted at ``data_root``.
    """

    data_root: Path = field(default_factory=default_data_root)

    def __post_init__(self):
        self.data_root = Path(self.data_root)

    # Raw photos, one file per image, class encoded in the filename prefix
    # (a_*.jpg, b_*.jpg, ...).
    @property
    def raw_dataset(self) -> Path:
        return self.data_root / "task_datasets"

    # Class-balanced copy, one folder per individual (A/, B/, ... H/).
    @property
    def balanced_dataset(self) -> Path:
        return self.data_root / "balanced_task_dataset"

    # SAM background-masked images.
    @property
    def masked_dataset(self) -> Path:
        return self.data_root / "masked_dataset"

    # Albumentations-augmented masked images.
    @property
    def augmented_dataset(self) -> Path:
        return self.data_root / "augmented_dataset"

    # Morphological (binary) masks derived from the masked dataset.
    @property
    def morph_dataset(self) -> Path:
        return self.data_root / "morp_dataset"

    # YOLO-format detection dataset (images/ + labels/ + data.yaml).
    @property
    def detection_dataset(self) -> Path:
        return self.data_root / "datasets"

    # Faces cropped out of detection images using YOLO labels.
    @property
    def cropped_faces(self) -> Path:
        return self.data_root / "cropped_faces"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_root / "embeddings"

    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.data_root / "outputs"


def load_config(path: str | Path) -> dict:
    """Load a YAML config file (see configs/default.yaml)."""
    with open(path) as f:
        return yaml.safe_load(f)
