"""Tests for Exercise 2.3: Proposal-distribution attack."""

# Pylint cannot infer callability of these PyTorch extension functions.
# pylint: disable=not-callable

import torch
from torch import Tensor

from src.adv.proposal_distribution_attack import ProposalDistributionAttack


def half_space_classifier(samples: Tensor) -> Tensor:
    """Classify points by the sign of their second coordinate."""
    return (samples[:, 1] < 0).to(torch.int64)


def toward_source(current: Tensor, generator: torch.Generator) -> Tensor:
    """Propose a step toward the source."""
    del generator
    return torch.tensor([0.0, 0.25], dtype=current.dtype, device=current.device)


def reject_non_adversarial(current: Tensor, generator: torch.Generator) -> Tensor:
    """Propose a step across the decision boundary."""
    del generator
    return torch.tensor([0.0, 3.0], dtype=current.dtype, device=current.device)


def increase_distance(current: Tensor, generator: torch.Generator) -> Tensor:
    """Propose a step away from the source."""
    del generator
    return torch.tensor([0.0, -0.5], dtype=current.dtype, device=current.device)


def outside_bounds(current: Tensor, generator: torch.Generator) -> Tensor:
    """Propose a step outside the configured bounds."""
    del generator
    return torch.tensor([0.0, -10.0], dtype=current.dtype, device=current.device)


def test_proposal_loop_accepts_adversarial_distance_improvement() -> None:
    """Accept an adversarial proposal that reduces the distance."""
    attack = ProposalDistributionAttack(
        half_space_classifier,
        toward_source,
        bounds=(-5.0, 5.0),
        max_iterations=1,
        max_queries=3,
        random_seed=4,
    )

    result = attack.attack(torch.tensor([0.0, 2.0]), torch.tensor([0.0, -2.0]))

    assert torch.equal(result.adversarial, torch.tensor([0.0, -1.75]))
    assert result.l2_distance < 4.0
    assert result.queries == 3


def test_proposal_loop_rejects_non_adversarial_candidate() -> None:
    """Reject a proposal that crosses the decision boundary."""
    attack = ProposalDistributionAttack(
        half_space_classifier,
        reject_non_adversarial,
        bounds=(-5.0, 5.0),
        max_iterations=1,
        max_queries=3,
        random_seed=5,
    )

    result = attack.attack(torch.tensor([0.0, 2.0]), torch.tensor([0.0, -2.0]))

    assert torch.equal(result.adversarial, torch.tensor([0.0, -2.0]))


def test_proposal_loop_rejects_distance_increase() -> None:
    """Reject an adversarial proposal that increases the distance."""
    attack = ProposalDistributionAttack(
        half_space_classifier,
        increase_distance,
        bounds=(-5.0, 5.0),
        max_iterations=1,
        max_queries=3,
        random_seed=5,
    )

    result = attack.attack(torch.tensor([0.0, 2.0]), torch.tensor([0.0, -2.0]))

    assert torch.equal(result.adversarial, torch.tensor([0.0, -2.0]))


def test_proposal_loop_rejects_out_of_bounds_without_querying() -> None:
    """Reject out-of-bounds proposals without a predictor query."""
    attack = ProposalDistributionAttack(
        half_space_classifier,
        outside_bounds,
        bounds=(-5.0, 5.0),
        max_iterations=1,
        max_queries=3,
        random_seed=5,
    )

    result = attack.attack(torch.tensor([0.0, 2.0]), torch.tensor([0.0, -2.0]))

    assert torch.equal(result.adversarial, torch.tensor([0.0, -2.0]))
    assert result.queries == 2
