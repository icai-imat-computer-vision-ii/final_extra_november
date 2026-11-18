"""Script to define the configuration of the project."""

from dataclasses import dataclass
from typing import Literal

import torch

USE_VECTORIZED = False


@dataclass(frozen=True)
class PathsConfig:
    """Class to define the paths used in the project."""

    dataset_root: str = "data"
    weights_dir: str = "weights/best_model.pt"
    training_figure: str = "images/results/training.png"
    examples: str = "images/results/test_samples.png"


@dataclass(frozen=True)
class DatasetConfig:
    """Class to define the configuration of the dataset."""

    image_size: tuple[int, int] = (96, 128)
    split_sizes: tuple[int, int, int] = (200, 75, 75)


@dataclass(frozen=True)
class ModelConfig:
    """Class to define the configuration of the model."""

    in_channels: int = 3
    out_channels: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and early-stopping settings."""

    # Normal training
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    optimizer = torch.optim.Adam
    # Early stopping
    patience: int = 5
    delta: float = 5e-3
    # Device
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    paths = PathsConfig()
    dataset = DatasetConfig()
    model = ModelConfig()
    training = TrainingConfig()


config = Config()
