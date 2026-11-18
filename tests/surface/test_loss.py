"""Check per-pixel cosine distance values, batch averaging and gradients."""

import torch

from final_extra.src.surface.dataset import NYUDepthV2
from final_extra.src.surface.loss import CosineDistance, VectorizedCosineDistance


def test_cosine_distances() -> None:
    """Ignore invalid targets and agree on values and gradients for valid pixels."""
    image, normals = NYUDepthV2._normalize_image_and_label(
        torch.full((3, 1, 2), 255.0),
        torch.tensor([[[0.0, 255.0]], [[0.0, 128.0]], [[0.0, 128.0]]]),
    )
    assert torch.all(image == 1)
    assert torch.all(normals[:, 0, 0] == 0)
    assert torch.allclose(normals[:, 0, 1].norm(), torch.tensor(1.0))
    predictions = (
        torch.tensor(
            [
                [2, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 2],
                [1, 2, 3],
                [1e-10, 0, 0],
                [0, 2, 0],
            ],
            dtype=torch.float64,
        )
        .reshape(2, 2, 2, 3)
        .permute(0, 3, 1, 2)
    )
    targets = (
        torch.tensor(
            [
                [3, 0, 0],
                [-2, 0, 0],
                [0, 3, 0],
                [0, 0, 0],
                [0, 0, 4],
                [2, 4, 6],
                [1, 0, 0],
                [0, 5, 0],
            ],
            dtype=torch.float64,
        )
        .reshape(2, 2, 2, 3)
        .permute(0, 3, 1, 2)
    )
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        gradients = []
        for criterion in (CosineDistance(), VectorizedCosineDistance()):
            prediction = predictions.to(device).detach().requires_grad_()
            target = targets.to(device)
            loss = criterion(prediction, target)
            expected = loss.new_tensor((2 + 1 + 0.99) / 7)
            assert loss.shape == () and torch.allclose(loss, expected)
            gradient = torch.autograd.grad(loss, prediction)[0]
            assert torch.isfinite(gradient).all()
            gradients.append(gradient)
            assert torch.all(gradient.permute(0, 2, 3, 1).reshape(-1, 3)[3] == 0)
            duplicated = criterion(
                prediction.repeat(2, 1, 1, 1), target.repeat(2, 1, 1, 1)
            )
            assert torch.allclose(duplicated, loss)
            invalid_loss = criterion(prediction, torch.zeros_like(target))
            assert invalid_loss.item() == 0
            invalid_gradient = torch.autograd.grad(invalid_loss, prediction)[0]
            assert torch.all(invalid_gradient == 0)
        assert torch.allclose(gradients[0], gradients[1])
