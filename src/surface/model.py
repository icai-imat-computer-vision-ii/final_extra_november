"""Script to define the U-Net architecture."""

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Class to construct a block of convolutions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Constructor of the class.

        Args:
            in_channels: Input channels of the block.
            out_channels: Output channels of the block.
        """

        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn_1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. It consists of two layers of: Conv + BN + ReLU.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].

        Returns:
            Output tensor. Dimensions: [batch, channels', height', width'].
        """

        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))

        return x


class EncoderBlock(nn.Module):
    """Class for an encoder block of the U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Constructor of the class.

        Args:
            in_channels: Input channels of the block.
            out_channels: Output channels of the block.
        """

        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].

        Returns:
            Tensors before (residual connection) and after the max pooling. Dimensions:
            [batch, channels', height', width'] and [batch, channels', height' // 2,
            width' // 2].
        """

        # TODO


class DecoderBlock(nn.Module):
    """Class for a decoder block of the U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Constructor of the class.

        Args:
            in_channels: Input channels of the block.
            out_channels: Output channels of the block.
        """

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].
            res: Residual connections of the corresponding layer to concatenate.
                Dimensions: [batch, channels, height, width].

        Returns:
            Output tensor. Dimensions: [batch, channels', height', width'].
        """

        # TODO


class UNet(nn.Module):
    """Class for the U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Constructor of the class. The variable `conv_sizes` defines the number of
        input and output channels for the encoders and decoders in a sequential order.
        For example, `[16, 32]` means we have one encoder with `16` output channels, a
        bottleneck with `32` output channels and a decoder with `16` output channels.

        The final convolution outputs are normalized across channels to produce
        surface normal direction vectors.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of outputs per pixel.
        """

        super().__init__()

        conv_sizes = [16, 32, 64]

        # Encoder
        self.encoder_1 = # TODO
        self.encoder_2 = # TODO

        # Bottleneck
        self.bottleneck = # TODO

        # Decoder
        self.decoder_1 = # TODO
        self.decoder_2 = # TODO

        # Classifier
        self.classifier = # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, in_channels, height, width].
                Height and width must be divisible by four for the skip connections.

        Returns:
            Normalized output. Dimensions: [batch, out_channels, height, width].
        """

        # TODO

        return F.normalize(x, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts the surface normals map of an image.

        Args:
            x: Input image. Dimensions: [in_channels, height, width].

        Returns:
            Predicted normal map. Dimensions: [out_channels, height, width].
        """

        self.eval()
        return self(x.unsqueeze(0)).squeeze(0)
