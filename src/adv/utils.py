"""Plot original and adversarial inputs with an amplified perturbation map."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor


def _image_tensor(image: Tensor, name: str) -> Tensor:
    """Convert a single image to a finite CPU tensor in CHW layout."""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"{name} must contain exactly one image")
        image = image.squeeze(0)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError(f"{name} must have shape (C, H, W) with C=1 or C=3")
    image = image.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} must contain only finite values")
    return image


def plot_attack_effect(
    original: Tensor,
    adversarial: Tensor,
    *,
    initial_adversarial: Tensor | None = None,
    value_range: tuple[float, float] = (0.0, 1.0),
    amplification: float = 8.0,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot the original, optional initial and final images, and their difference."""
    lower, upper = value_range
    if not lower < upper:
        raise ValueError("value_range lower bound must be below its upper bound")
    if amplification <= 0:
        raise ValueError("amplification must be positive")

    source = _image_tensor(original, "original")
    adversarial_image = _image_tensor(adversarial, "adversarial")
    if source.shape != adversarial_image.shape:
        raise ValueError("original and adversarial must have equal shapes")
    initial_image = (
        _image_tensor(initial_adversarial, "initial_adversarial")
        if initial_adversarial is not None
        else None
    )
    if initial_image is not None and source.shape != initial_image.shape:
        raise ValueError("original and initial_adversarial must have equal shapes")

    span = upper - lower
    source_display = ((source - lower) / span).clamp(0, 1)
    adversarial_display = ((adversarial_image - lower) / span).clamp(0, 1)
    image_panels = [("Original", source_display)]
    if initial_image is not None:
        initial_display = ((initial_image - lower) / span).clamp(0, 1)
        image_panels.append(("Initial adversarial", initial_display))
    image_panels.append(("Adversarial", adversarial_display))
    difference = (adversarial_image - source).abs().mean(dim=0)
    difference_display = (difference * amplification / span).clamp(0, 1)
    distance = torch.linalg.vector_norm(adversarial_image - source).item()

    figure, axes = plt.subplots(
        1,
        len(image_panels) + 1,
        figsize=(4 * (len(image_panels) + 1), 4),
        constrained_layout=True,
    )
    for axis, (title, image) in zip(axes, image_panels):
        if image.shape[0] == 1:
            axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(title)
    axes[-1].imshow(difference_display.numpy(), cmap="inferno", vmin=0, vmax=1)
    axes[-1].set_title(f"Mean |difference| ×{amplification:g}")
    figure.suptitle(f"L2 perturbation: {distance:.4g}")
    for axis in axes:
        axis.axis("off")

    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return figure


_IMAGE_DIR = Path(__file__).resolve().parents[2] / "images/coco"


def load_coco_images() -> Tensor:
    """Load images 0-3 as RGB float tensors shaped (4, 3, 224, 224) in [0, 1]."""
    images = []
    for index in range(4):
        with Image.open(_IMAGE_DIR / f"{index}.jpg") as image:
            resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            pixels = np.array(resized, dtype=np.float32)
        images.append(torch.from_numpy(pixels).permute(2, 0, 1) / 255.0)
    return torch.stack(images)
