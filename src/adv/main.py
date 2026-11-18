"""Run a Boundary Attack on the supplied COCO images."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchvision.models import ResNet18_Weights, resnet18

from .boundary_attack import BoundaryAttack
from .utils import load_coco_images, plot_attack_effect


def main() -> None:
    """Run the attack and save a comparison of the original and adversarial images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, choices=range(4), default=0)
    parser.add_argument("--initial", type=int, choices=range(4), default=1)
    parser.add_argument("--max-queries", type=int, default=2500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        f"images/results_adv/boundary_{args.source}_{args.initial}.png"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def predictor(images: Tensor) -> Tensor:
        return model((images - mean) / std).argmax(dim=1)

    images = load_coco_images()
    original, initial = images[args.source], images[args.initial]
    attack = BoundaryAttack(
        predictor,
        bounds=(0.0, 1.0),
        device=device,
        max_queries=args.max_queries,
        random_seed=7,
    )
    result = attack.attack(original, initial)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_attack_effect(
        original,
        result.adversarial,
        initial_adversarial=initial,
        save_path=output,
    )
    plt.close(figure)
    categories = weights.meta["categories"]
    print(f"Original: {categories[result.original_label]}")
    print(f"Adversarial: {categories[result.adversarial_label]}")
    print(f"L2 distance: {result.l2_distance:.4f}; queries: {result.queries}")
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()
