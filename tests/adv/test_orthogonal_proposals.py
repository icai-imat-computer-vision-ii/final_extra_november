"""Tests for Exercise 2.1: Orthogonal proposals."""

# Pylint cannot infer callability of these PyTorch extension functions.
# pylint: disable=not-callable

import torch

from src.adv.boundary_attack import orthogonal_proposals


def test_orthogonal_directions_and_sphere_projection() -> None:
    """Check tangency and sphere-distance preservation."""
    source = torch.tensor([0.0, 0.0, 0.0])
    current = torch.tensor([1.0, 2.0, -1.0])
    directions, candidates = orthogonal_proposals(
        source, current, 0.2, 8, torch.Generator().manual_seed(7)
    )

    dot_products = directions.flatten(start_dim=1) @ (current - source)
    distances = torch.linalg.vector_norm(candidates - source, dim=1)

    assert torch.allclose(dot_products, torch.zeros_like(dot_products), atol=1e-6)
    assert torch.all(torch.linalg.vector_norm(directions, dim=1) > 0)
    assert torch.allclose(
        distances,
        torch.linalg.vector_norm(current - source).expand_as(distances),
        atol=1e-6,
    )
