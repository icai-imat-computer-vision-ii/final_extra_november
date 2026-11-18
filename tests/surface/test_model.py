"""Check configured dataset sizes and U-Net input channels."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from final_extra.src.surface import main
from final_extra.src.surface.dataset import NYUDepthV2


def test_model_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use nondefault settings and produce normalized three-channel predictions."""
    for split in ("train", "val", "test"):
        image_dir = tmp_path / split / "images"
        image_dir.mkdir(parents=True)
        for index in range(5):
            (image_dir / f"{index}.png").touch()
    monkeypatch.setattr(
        main,
        "config",
        SimpleNamespace(
            paths=SimpleNamespace(dataset_root=str(tmp_path)),
            dataset=SimpleNamespace(image_size=(8, 12), split_sizes=(3, 2, 8)),
            model=SimpleNamespace(in_channels=1, out_channels=3),
            training=SimpleNamespace(device="cpu"),
        ),
    )
    datasets = main.load_datasets()
    for subset, split in zip(datasets, ("train", "val", "test")):
        dataset = subset.dataset
        assert isinstance(dataset, NYUDepthV2)
        assert dataset.image_size == (8, 12)
        assert dataset.path == f"{tmp_path}/{split}"
    assert [len(dataset) for dataset in datasets] == [3, 2, 5]
    assert [dataset.indices for dataset in datasets] == [
        dataset.indices for dataset in main.load_datasets()
    ]
    for dataset in datasets:
        assert len(set(dataset.indices)) == len(dataset)
        assert all(0 <= index < 5 for index in dataset.indices)
    model = main.create_model().eval()
    assert model.encoder_1.conv.conv_1.in_channels == 1
    image = torch.rand(1, 8, 12)
    prediction = model.predict(image)
    assert prediction.shape == (3, 8, 12)
    assert not prediction.requires_grad
    assert torch.isfinite(prediction).all()
    assert torch.allclose(prediction.norm(dim=0), torch.ones(8, 12), atol=1e-6)

    model.train()
    outputs = model(torch.rand(2, 1, 8, 12))
    assert outputs.shape == (2, 3, 8, 12)
    outputs[:, 0].mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
