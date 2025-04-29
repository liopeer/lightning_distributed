from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import Dataset, DataLoader
from neuralop.data.datasets.spherical_swe import SphericalSWEDataset
from model import FNOModel


def get_trainval_datasets(dims: tuple[int, int], num_examples: int, train_val_split: float = 0.8) -> tuple[Dataset]:

    train_ds = SphericalSWEDataset(
        dims=dims,
        num_examples=num_examples,
        device=torch.device("cpu"),
    )

    val_ds = SphericalSWEDataset(
        dims=dims,
        num_examples=num_examples,
        device=torch.device("cpu"),
    )

    return train_ds, val_ds


def get_trainval_dataloaders(
    train_ds: Dataset, val_ds: Dataset, batch_size: int
) -> tuple[DataLoader]:
    train_dl = DataLoader(
        dataset=train_ds, batch_size=batch_size, shuffle=True
    )

    val_dl = DataLoader(
        dataset=val_ds, batch_size=batch_size, shuffle=False
    )

    return train_dl, val_dl


def setup_trainer(devices: int | str = "auto") -> Trainer:
    trainer = Trainer(
        accelerator="mps",
        devices=devices,
        max_epochs=10,
    )

    return trainer

def main() -> None:
    # Define the model
    model = FNOModel("sfno", {"n_modes": [24, 24], "hidden_channels": 256, "n_layers": 5, "in_channels": 3, "out_channels": 3})

    # Define the dataset parameters
    dims = (128, 128)
    num_examples = 1000
    batch_size = 32

    # Get the datasets and dataloaders
    train_ds, val_ds = get_trainval_datasets(dims, num_examples)
    train_dl, val_dl = get_trainval_dataloaders(train_ds, val_ds, batch_size)

    # Setup the trainer
    trainer = setup_trainer()

    # Train the model
    trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    main()