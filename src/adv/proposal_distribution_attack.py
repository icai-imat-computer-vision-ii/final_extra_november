"""Student template for the proposal-distribution attack."""

from collections.abc import Callable

import torch
from torch import Tensor

from src.adv.boundary_attack import BoundaryAttackResult, Predictor

ProposalDistribution = Callable[[Tensor, torch.Generator], Tensor]


class ProposalDistributionAttack:
    """Apply accept/reject search using a caller-supplied perturbation sampler.

    The sampler returns ``eta`` (a perturbation), not the proposed candidate.
    """

    def __init__(
        self,
        predictor: Predictor,
        proposal_distribution: ProposalDistribution,
        *,
        bounds: tuple[float, float] = (0.0, 255.0),
        device: torch.device | str | None = None,
        max_iterations: int = 1000,
        max_queries: int = 10000,
        distance_tolerance: float = 1e-3,
        require_improvement: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """Configure the predictor, proposal sampler, and search limits."""
        self.predictor = predictor
        self.proposal_distribution = proposal_distribution
        self.lower_bound, self.upper_bound = bounds
        self.device = torch.device(device) if device is not None else None
        self.max_iterations = max_iterations
        self.max_queries = max_queries
        self.distance_tolerance = distance_tolerance
        self.require_improvement = require_improvement
        self.random_seed = random_seed

    def attack(
        self,
        original: Tensor,
        initial_adversarial: Tensor,
        target_label: int | None = None,
    ) -> BoundaryAttackResult:
        """Search using a proposal sampler from an adversarial starting input.

        TODO:
        1. Move the supplied inputs to the selected device and classify them.
           Import and use ``_validate_initial_labels`` from ``.boundary_attack``
           to check the initial adversarial criterion.
        2. Create and seed a ``torch.Generator`` on the attack device.
        3. For each iteration, get a perturbation from the callback, add it to
           the current candidate, and reject candidates outside ``bounds``.
        4. Query the classifier and accept candidates that satisfy the requested
           adversarial criterion. By default, also require a strict L2 decrease;
           when ``require_improvement`` is false, use adversarial-only acceptance.
        5. Stop at the iteration, query, or distance limit and return the shared
           result type with accurate query and iteration counts.

        Args:
            original: Unperturbed input.
            initial_adversarial: Starting input satisfying the selected criterion.
            target_label: Required adversarial class, or None for any class
                other than the original prediction.

        Returns:
            The final adversarial input, labels, query and iteration counts,
            and L2 distance as a ``BoundaryAttackResult``.
        """
        raise NotImplementedError("TODO: implement the proposal-distribution loop")
