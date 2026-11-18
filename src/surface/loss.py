"""Script to define the cosine distance for the loss function."""

import torch
from torch import nn


class CosineDistance(nn.Module):
    """Class to define the cosine distance."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            y_pred: Predicted labels. Dimensions: [batch, channels, height, width].
            y_true: True labels. Dimensions: [batch, channels, height, width].

        Returns:
            Mean cosine distance over valid pixels; zero targets are ignored.
        """

        # TODO


class VectorizedCosineDistance(nn.Module):
    """Class to define the vectorized cosine distance."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            y_pred: Predicted labels. Dimensions: [batch, channels, height, width].
            y_true: True labels. Dimensions: [batch, channels, height, width].

        Returns:
            Mean cosine distance over valid pixels; zero targets are ignored.
        """

        # TODO
