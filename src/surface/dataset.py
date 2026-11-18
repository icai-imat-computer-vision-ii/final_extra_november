"""Load numbered NYU Depth V2 image and surface normal pairs."""

import os
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


class NYUDepthV2(Dataset):
    """RGB images and normal maps from a prepared train, val or test split."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        image_size: tuple[int, int] = (192, 256),
    ) -> None:
        """Constructor of the class.

        Args:
            root: Path where the dataset is stored.
            split: Split of the dataset.
            image_size: (Height, width), both positive multiples of 4.

        Raises:
            ValueError: If either dimension is not a positive multiple of 4.
        """

        if any(size <= 0 or size % 4 for size in image_size):
            raise ValueError("Image height and width must be positive multiples of 4.")

        self.path = f"{root}/{split}"
        self.image_size = image_size

    def __len__(self) -> int:
        """Count the numbered image pairs.

        Returns:
            Length of the dataset.
        """

        return len(os.listdir(f"{self.path}/images"))

    @staticmethod
    def _normalize_image_and_label(
        image: torch.Tensor, surface_normals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalizes the image and its label.

        Args:
            image: Image. Dimensions: [3, height, width].
            surface_normals: Surface normals. Dimensions: [3, height, width].

        Returns:
            RGB image in [0, 1] and unit normals, both shaped [3, height, width].
            Invalid normal pixels remain zero.
        """

        image /= 255  # between 0 and 1

        valid = surface_normals.any(dim=0, keepdim=True)  # forget about 0s
        surface_normals = (
            surface_normals / 127.5 - 1
        ) * valid  # normalize between -1 and 1
        surface_normals = F.normalize(surface_normals, dim=0)  # make norm equal to 1

        return image, surface_normals

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtains the image and label (surface normals) corresponding to the index.

        Args:
            index: Index within the dataset.

        Returns:
            Image and its label. Dimensions: [3, channels, height, width] both.
        """

        with (
            Image.open(f"{self.path}/images/{index}.png") as image_file,
            Image.open(f"{self.path}/normals/{index}.png") as normals_file,
        ):
            image = image_file.convert("RGB")
            surface_normals = normals_file.convert("RGB")

            height, width = self.image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            surface_normals = surface_normals.resize(
                (width, height), Image.Resampling.NEAREST
            )

            rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
            normals = (
                torch.from_numpy(np.array(surface_normals)).permute(2, 0, 1).float()
            )

        return self._normalize_image_and_label(rgb, normals)
