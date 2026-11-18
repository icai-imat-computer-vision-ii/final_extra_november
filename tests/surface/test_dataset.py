"""Check loading, resizing and decoding a pair of known samples."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from final_extra.src.surface.dataset import NYUDepthV2


def test_dataset(tmp_path: Path) -> None:
    """Check RGBA pairing, signed unit normals, resizing and invalid pixels."""
    for directory in ("images", "normals"):
        (tmp_path / "train" / directory).mkdir(parents=True)
    for index, color in enumerate(([255, 0, 0, 255], [0, 255, 0, 255])):
        raw_image = np.array([[color, [0, 0, 255, 255]]], dtype=np.uint8)
        raw_normals = np.array(
            [[[255 * index, 128, 128, 255], [0, 0, 0, 255]]], dtype=np.uint8
        )
        Image.fromarray(raw_image).save(tmp_path / "train" / "images" / f"{index}.png")
        Image.fromarray(raw_normals).save(
            tmp_path / "train" / "normals" / f"{index}.png"
        )

    for size in ((4, 8), (192, 256)):
        dataset = NYUDepthV2(str(tmp_path), "train", image_size=size)
        assert len(dataset) == 2
        height, width = size
        for index in range(2):
            image, normals = dataset[index]
            assert image.shape == normals.shape == (3, height, width)
            assert image.dtype == normals.dtype == torch.float32
            expected_rgb = torch.tensor([1.0 - index, float(index), 0.0])
            assert torch.equal(image[:, 0, 0], expected_rgb)
            assert normals[0, 0, 0] * (2 * index - 1) > 0.99
            assert torch.allclose(
                normals[:, :, : width // 2].norm(dim=0),
                torch.ones(height, width // 2),
            )
            assert torch.all(normals[:, :, width // 2 :] == 0)

    for size in ((190, 256), (192, 255), (0, 256), (-4, 256)):
        with pytest.raises(ValueError, match="positive multiples of 4"):
            NYUDepthV2(str(tmp_path), "train", image_size=size)
