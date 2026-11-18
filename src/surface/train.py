"""Script to train the point cloud classifier."""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from final_extra.src.surface.config import USE_VECTORIZED, TrainingConfig
from final_extra.src.surface.early_stopping import EarlyStopping
from final_extra.src.surface.loss import CosineDistance, VectorizedCosineDistance
from final_extra.src.surface.model import UNet


class Trainer:
    """Class to train the surface normals predictor."""

    def __init__(self, model: UNet, training_config: TrainingConfig) -> None:
        """Constructor of the class.

        Args:
            model: Model to train.
            training_config: Configuration for the training.
        """

        self.model = model.to(training_config.device)
        self.optimizer = training_config.optimizer(
            self.model.parameters(), training_config.learning_rate
        )
        self.early_stopping = EarlyStopping(
            training_config.patience, training_config.delta
        )
        self.device = training_config.device
        self.criterion = (
            VectorizedCosineDistance() if USE_VECTORIZED else CosineDistance()
        )
        self.training_metrics: dict[str, list[float]] = {"train": [], "val": []}

    def _train_one_epoch(self, train_loader: DataLoader) -> None:
        """Trains the model one epoch.

        Args:
            train_loader: Train loader.
        """

        # TODO

        self.training_metrics["train"].append(total_loss / len(train_loader))

    @torch.no_grad()
    def _valid_one_epoch(self, val_loader: DataLoader) -> None:
        """Validates the model one epoch.

        Args:
            val_loader: Validation loader.
        """

        # TODO

        self.training_metrics["val"].append(total_loss / len(val_loader))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        *,
        weights_path: str,
        figure_path: str,
    ) -> UNet:
        """Trains the model.

        Args:
            train_loader: Train loader.
            val_loader: Validation loader.
            epochs: Number of epochs to train.
            weights_path: Path where the weights of the model will be saved.
            figure_path: Path where the figure of the training evolution will be saved.

        Returns:
            Trained model with the best weights.
        """

        # TODO (start training in epoch 1)

        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.save_training_figure(figure_path)

        return self.model

    def _print_training_progress(self, epoch: int, epochs: int) -> None:
        """Prints on screen the current train and validation losses and accuracies.

        Args:
            epoch: Current epoch.
            epochs: Total number of epochs.
        """

        save_every = max(1, epochs // 5)

        if not (epoch == 1 or epoch == epochs or epoch % save_every == 0):
            return

        last_loss_train = self.training_metrics["train"][-1]
        last_loss_valid = self.training_metrics["val"][-1]

        print(
            f"Epoch: {epoch}. Loss train: {last_loss_train:.3f} | Loss validation: "
            f"{last_loss_valid:.3f}"
        )

    def save_training_figure(self, path: str) -> None:
        """Saves the evolution of the losses and accuracies.

        Args:
            path: Path where the figure is saved.
        """

        epochs = range(1, len(self.training_metrics["train"]) + 1)
        figure, axs = plt.subplots(1, 1, figsize=(12, 5))

        axs.plot(epochs, self.training_metrics["train"], label="Train")
        axs.plot(epochs, self.training_metrics["val"], label="Validation")
        axs.set_title("Evolution of the loss")

        axs.set_xlabel("Epoch")
        axs.set_ylabel("Cosine Distance")
        axs.grid()
        axs.legend()

        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
