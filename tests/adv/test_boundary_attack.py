"""Tests for Exercise 2.2: Boundary Attack."""

# Pylint cannot infer callability of these PyTorch extension functions.
# pylint: disable=not-callable

import torch
from torch import Tensor

from src.adv.boundary_attack import BoundaryAttack


def half_space_classifier(samples: Tensor) -> Tensor:
    """Classify points by the sign of their second coordinate."""
    return (samples[:, 1] < 0).to(torch.int64)


def test_boundary_attack_reduces_distance_and_stays_adversarial() -> None:
    """Check progress, adversarial validity, and bounds."""
    attack = BoundaryAttack(
        half_space_classifier,
        bounds=(-5.0, 5.0),
        max_iterations=40,
        max_queries=500,
        batch_size=8,
        epsilon=0.2,
        random_seed=7,
    )

    original = torch.tensor([0.0, 2.0])
    initial = torch.tensor([0.0, -2.0])
    result = attack.attack(original, initial)

    assert result.original_label == 0
    assert result.adversarial_label == 1
    assert half_space_classifier(result.adversarial.unsqueeze(0)).item() == 1
    assert torch.linalg.vector_norm(original - result.adversarial) < 4.0
    assert torch.all(result.adversarial >= -5.0)
    assert torch.all(result.adversarial <= 5.0)


def test_boundary_attack_counts_queries_per_sample() -> None:
    """Count each sample in a batch as a separate query."""
    observed_queries = 0

    def predictor(samples: Tensor) -> Tensor:
        nonlocal observed_queries
        observed_queries += len(samples)
        return half_space_classifier(samples)

    attack = BoundaryAttack(
        predictor,
        bounds=(-5.0, 5.0),
        max_iterations=5,
        max_queries=3,
        batch_size=8,
        random_seed=3,
    )
    result = attack.attack(torch.tensor([0.0, 2.0]), torch.tensor([0.0, -2.0]))

    assert observed_queries == 3
    assert result.queries == observed_queries
