"""Student template for the decision-based L2 Boundary Attack."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

Predictor = Callable[[Tensor], Tensor]


def _within_bounds(sample: Tensor, lower: float, upper: float) -> bool:
    """Return whether every value lies within the scalar bounds."""
    return bool(torch.all(sample >= lower) and torch.all(sample <= upper))


def _distance(first: Tensor, second: Tensor) -> float:
    """Return the L2 distance between two tensors."""
    return torch.linalg.vector_norm(first - second).item()


def _predict_labels(predictor: Predictor, samples: Tensor) -> Tensor:
    """Run the predictor and convert class scores to labels when necessary."""
    with torch.inference_mode():
        predictions = predictor(samples)
    if predictions.ndim == 2:
        predictions = predictions.argmax(dim=1)
    return predictions.to(device=samples.device)


def _is_adversarial(label: int, original_label: int, target_label: int | None) -> bool:
    """Check the targeted or untargeted adversarial criterion."""
    return (
        label == target_label if target_label is not None else label != original_label
    )


def _validate_initial_labels(
    original_label: int, initial_label: int, target_label: int | None
) -> None:
    """Require the initial point to satisfy the requested attack criterion."""
    if target_label is not None and target_label == original_label:
        raise ValueError("target_label must differ from the original label")
    if target_label is not None and initial_label != target_label:
        raise ValueError("initial_adversarial must have the target_label")
    if target_label is None and initial_label == original_label:
        raise ValueError("initial_adversarial must be classified as adversarial")


def orthogonal_proposals(
    source: Tensor,
    current: Tensor,
    delta: float,
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Generate orthogonal directions and sphere-projected candidates.

    TODO: sample random directions with ``generator``, remove each direction's
    component parallel to ``current - source``, scale the tangent directions,
    and project the candidates back onto the sphere centered at ``source``.

    Args:
        source: Original input at the center of the sphere.
        current: Current adversarial input, distinct from the source.
        delta: Scale of the tangent step relative to the current L2 distance.
        count: Number of proposals to generate.
        generator: Random generator used to sample directions.

    Returns:
        Tangent directions and sphere-projected candidates, each with a leading
        dimension of size ``count``. Candidates are returned before clipping.
    """
    raise NotImplementedError("TODO: implement orthogonal proposals")


@dataclass(frozen=True)
class BoundaryAttackResult:
    """Outcome and query statistics for an attack run."""

    adversarial: Tensor
    original_label: int
    adversarial_label: int
    queries: int
    iterations: int
    l2_distance: float


class BoundaryAttack:
    """Reduce L2 distance using only predicted labels and orthogonal proposals."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        bounds: tuple[float, float] = (0.0, 255.0),
        device: torch.device | str | None = None,
        max_iterations: int = 1000,
        max_queries: int = 10000,
        batch_size: int = 10,
        max_forward_trials: int = 20,
        delta: float = 0.1,
        epsilon: float = 0.01,
        distance_tolerance: float = 1e-3,
        random_seed: int | None = None,
    ) -> None:
        """Configure the predictor, search parameters, and query budget."""
        self.predictor = predictor
        self.lower_bound, self.upper_bound = bounds
        self.device = torch.device(device) if device is not None else None
        self.max_iterations = max_iterations
        self.max_queries = max_queries
        self.batch_size = batch_size
        self.max_forward_trials = max_forward_trials
        self.initial_delta = delta
        self.initial_epsilon = epsilon
        self.distance_tolerance = distance_tolerance
        self.random_seed = random_seed
        self._queries = 0
        self._generator = torch.Generator()

    def _predict_labels(self, samples: Tensor) -> Tensor:
        """Predict labels while counting one query per sample."""
        if self._queries + len(samples) > self.max_queries:
            raise RuntimeError("internal query budget exceeded")
        self._queries += len(samples)
        return _predict_labels(self.predictor, samples)

    def _orthogonal_proposals(
        self,
        source: Tensor,
        current: Tensor,
        delta: float,
        count: int,
    ) -> Tensor:
        """Generate projected candidates and apply the configured input bounds."""
        _, candidates = orthogonal_proposals(
            source, current, delta, count, self._generator
        )
        return torch.clamp(candidates, min=self.lower_bound, max=self.upper_bound)

    def _step_toward_source(
        self,
        source: Tensor,
        adversarial: Tensor,
        current_label: int,
        original_label: int,
        target_label: int | None,
        epsilon: float,
    ) -> tuple[Tensor, int, float]:
        """Try inward steps until one stays adversarial or the budget runs out.

        TODO: try up to ``max_forward_trials`` bounded steps toward the source,
        querying through ``self._predict_labels`` without exceeding the budget.
        Accept a step only if it remains adversarial; increase ``epsilon`` on
        success and decrease it after each unsuccessful trial.

        Args:
            source: Original input toward which to move.
            adversarial: Current adversarial input.
            current_label: Label of the current adversarial input.
            original_label: Label of the original input.
            target_label: Required adversarial class, or None for any other class.
            epsilon: Current source-directed step size.

        Returns:
            The adversarial input, its label, and the updated step size.
        """
        raise NotImplementedError("TODO: implement the step toward the source")

    def attack(
        self,
        original: Tensor,
        initial_adversarial: Tensor,
        target_label: int | None = None,
    ) -> BoundaryAttackResult:
        """Reduce the distance from an adversarial input to the original.

        TODO:
        1. Move the supplied inputs to the selected device and seed a generator.
        2. Classify source and initialization; use ``_validate_initial_labels``.
        3. Propose a batch of tangent candidates and accept an adversarial one.
        4. Adapt ``delta`` from proposal success, then call
           ``self._step_toward_source`` to try an inward step.
        5. Respect the query and iteration budgets; report counts and L2 distance.

        Args:
            original: Unperturbed input.
            initial_adversarial: Starting input satisfying the selected criterion.
            target_label: Required adversarial class, or None for any class
                other than the original prediction.

        Returns:
            The final adversarial input, labels, query and iteration counts,
            and L2 distance as a ``BoundaryAttackResult``.
        """
        raise NotImplementedError("TODO: implement the Boundary Attack loop")
