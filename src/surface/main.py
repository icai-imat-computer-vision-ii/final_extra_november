"""Script to train the model and evaluate in the dataset."""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from final_extra.src.surface.config import USE_VECTORIZED, config
from final_extra.src.surface.dataset import NYUDepthV2
from final_extra.src.surface.loss import CosineDistance, VectorizedCosineDistance
from final_extra.src.surface.model import UNet
from final_extra.src.surface.train import Trainer


def load_datasets() -> tuple[Subset, Subset, Subset]:
    """Loads fixed random subsets of the train, validation and test splits.

    Returns:
        Train, validation and test datasets.
    """

    train_dataset = NYUDepthV2(
        config.paths.dataset_root, "train", config.dataset.image_size
    )
    val_dataset = NYUDepthV2(
        config.paths.dataset_root, "val", config.dataset.image_size
    )
    test_dataset = NYUDepthV2(
        config.paths.dataset_root, "test", config.dataset.image_size
    )

    generator = torch.Generator().manual_seed(42)
    subsets = [
        Subset(
            dataset,
            torch.randperm(len(dataset), generator=generator)[:size].tolist(),
        )
        for dataset, size in zip(
            (train_dataset, val_dataset, test_dataset), config.dataset.split_sizes
        )
    ]
    return subsets[0], subsets[1], subsets[2]


def create_model() -> UNet:
    """Creates the model on the configured device.

    Returns:
        Surface normals predictor.
    """

    model = UNet(
        in_channels=config.model.in_channels, out_channels=config.model.out_channels
    )
    return model.to(config.training.device)


def train_model(train_dataset: Subset, val_dataset: Subset) -> UNet:
    """Trains a new model and returns the best validation checkpoint.

    Args:
        train_dataset: Train dataset.
        val_dataset: Validation dataset.

    Returns:
        Trained cloud point classifier.
    """

    model = create_model()
    trainer = Trainer(model, config.training)
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False
    )

    return trainer.fit(
        train_loader,
        val_loader,
        epochs=config.training.epochs,
        weights_path=config.paths.weights_dir,
        figure_path=config.paths.training_figure,
    )


@torch.no_grad()
def evaluate_model(test_dataset: Subset) -> None:
    """Loads the best weights into a new model and prints test loss and accuracy.

    Args:
        test_dataset: Test dataset.
    """

    model = create_model()
    model.load_state_dict(
        torch.load(
            config.paths.weights_dir,
            map_location=config.training.device,
            weights_only=True,
        )
    )
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    criterion = VectorizedCosineDistance() if USE_VECTORIZED else CosineDistance()
    total_loss = 0.0

    for cloud_point, label in test_loader:
        cloud_point = cloud_point.to(config.training.device)
        label = label.to(config.training.device)
        outputs = model(cloud_point)
        total_loss += criterion(outputs, label).item()

    print(f"Test loss: {total_loss / len(test_loader):.3f}")


@torch.no_grad()
def draw_test_samples(test_dataset: Subset, num_samples: int = 5) -> None:
    """Save and display RGB images alongside true and predicted surface normals.

    Args:
        test_dataset: Test dataset.
        num_samples: Number of test images to draw.
    """

    model = create_model()
    model.load_state_dict(
        torch.load(
            config.paths.weights_dir,
            map_location=config.training.device,
            weights_only=True,
        )
    )
    model.eval()
    indices = torch.randperm(len(test_dataset))[:num_samples].tolist()
    figure, axes = plt.subplots(
        num_samples, 3, figsize=(12, 3 * num_samples), squeeze=False
    )

    for row, sample_index in enumerate(indices):
        image, label = test_dataset[sample_index]
        prediction = model.predict(image.to(config.training.device)).cpu()
        # Map signed normal components to RGB colors for display.
        panels = (image, (label + 1) / 2, (prediction + 1) / 2)
        titles = (f"RGB image ({sample_index})", "True normals", "Predicted normals")
        for axis, panel, title in zip(axes[row], panels, titles):
            axis.imshow(panel.cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            axis.set_title(title)
            axis.set_axis_off()

    figure.tight_layout()
    figure.savefig(config.paths.examples)
    plt.show(block=True)
    plt.close(figure)


def main() -> None:
    """Creates the datasets, trains the model and evaluates it."""

    train, val, test = load_datasets()
    train_model(train, val)

    evaluate_model(test)
    draw_test_samples(test)


if __name__ == "__main__":
    main()
