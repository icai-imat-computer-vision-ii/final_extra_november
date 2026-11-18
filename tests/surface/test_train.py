"""Tests for the training."""

import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.surface.config import USE_VECTORIZED, TrainingConfig
from src.surface.loss import CosineDistance, VectorizedCosineDistance
from src.surface.model import UNet
from src.surface.train import Trainer

IN_CHANNELS = 3
OUT_CHANNELS = 3
NUM_IMAGES = 3
IMAGE_SIZE = 8
EPOCHS = 20
TRAINING_CONFIG = TrainingConfig(device="cpu", batch_size=2, patience=2, delta=1.0)


@pytest.fixture
def trainer() -> Trainer:
    """Creates a deterministic CPU trainer with a tiny model."""

    model = UNet(IN_CHANNELS, OUT_CHANNELS)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        assert model.classifier.bias is not None
        model.classifier.bias[0] = 1.0

    return Trainer(model, TRAINING_CONFIG)


@pytest.fixture
def loader() -> DataLoader:
    """Creates image/normal batches, including an incomplete final batch."""

    images = torch.ones(NUM_IMAGES, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.zeros(NUM_IMAGES, OUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    labels[:, 2] = 1.0
    dataset = TensorDataset(images, labels)

    return DataLoader(dataset, batch_size=TRAINING_CONFIG.batch_size)


def test_init(trainer: Trainer) -> None:
    """Checks configuration, device, optimizer, and empty metrics."""

    assert trainer.device == TRAINING_CONFIG.device
    assert all(
        parameter.device.type == trainer.device
        for parameter in trainer.model.parameters()
    )
    assert isinstance(trainer.optimizer, TRAINING_CONFIG.optimizer)
    assert isinstance(
        trainer.criterion,
        VectorizedCosineDistance if USE_VECTORIZED else CosineDistance,
    )
    assert trainer.early_stopping.patience == TRAINING_CONFIG.patience
    assert trainer.early_stopping.delta == TRAINING_CONFIG.delta
    assert trainer.training_metrics == {"train": [], "val": []}


def test_train_one_epoch(trainer: Trainer, loader: DataLoader) -> None:
    """Checks training records loss and updates weights with incomplete batches."""

    for epoch in range(1, 3):
        before = {
            name: value.clone() for name, value in trainer.model.named_parameters()
        }
        trainer._train_one_epoch(loader)

        assert trainer.model.training

        metrics = trainer.training_metrics["train"]
        assert len(metrics) == epoch
        assert 0 <= metrics[-1] <= 2

        assert any(
            not torch.equal(before[name], value)
            for name, value in trainer.model.named_parameters()
        )


def test_valid_one_epoch(trainer: Trainer, loader: DataLoader) -> None:
    """Checks validation records the expected metrics without changing weights."""

    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    trainer._valid_one_epoch(loader)

    assert not trainer.model.training

    metrics = trainer.training_metrics["val"]
    assert len(metrics) == 1

    # The fixture predicts X-axis normals while the targets point along Z.
    assert torch.isclose(torch.tensor(metrics[0]), torch.tensor(1.0))

    assert all(
        torch.equal(before[name], value)
        for name, value in trainer.model.state_dict().items()
    )


def test_fit(trainer: Trainer, loader: DataLoader, tmp_path) -> None:
    """Checks early stopping, restoring the checkpoint, and saving the figure."""

    weights_path = f"{tmp_path}/best.pt"
    figure_path = f"{tmp_path}/training.png"
    model = trainer.fit(
        loader,
        loader,
        EPOCHS,
        weights_path=weights_path,
        figure_path=figure_path,
    )
    saved = torch.load(weights_path, weights_only=True)

    assert model is trainer.model
    assert len(trainer.training_metrics["train"]) == TRAINING_CONFIG.patience + 1
    assert trainer.early_stopping.apply_early_stop
    assert all(
        torch.equal(saved[name], value) for name, value in model.state_dict().items()
    )
    assert os.path.isfile(figure_path)


def test_save_training_figure(trainer: Trainer, tmp_path) -> None:
    """Checks the figure is saved."""

    for metrics in trainer.training_metrics.values():
        metrics.append(0.5)
    figure_path = f"{tmp_path}/training.png"
    trainer.save_training_figure(figure_path)

    assert os.path.isfile(figure_path)
