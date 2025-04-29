from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import Dataset, DataLoader
from neuralop.data.datasets.spherical_swe import SphericalSWEDataset
import torch.distributed as dist
from torch.distributed import TCPStore
from lightning.pytorch.plugins.environments import (
    SLURMEnvironment,
    LightningEnvironment,
)
from model import FNOModel
import os
import datetime
import re


def get_trainval_datasets(
    dims: tuple[int, int], num_examples: int, train_val_split: float = 0.8
) -> tuple[Dataset]:
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
        dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=8
    )

    val_dl = DataLoader(
        dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=8
    )

    return train_dl, val_dl


def setup_trainer(devices: int | str = "auto") -> Trainer:
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        max_epochs=10,
    )

    return trainer


def get_world_size() -> int:
    if SLURMEnvironment.detect():
        return int(os.environ.get("SLURM_NTASKS"))
    else:
        return os.environ.get("WORLD_SIZE", 1)


def get_global_rank() -> int | None:
    """Get the global rank of the current process.
    Copied from https://github.com/Lightning-AI/pytorch-lightning/blob/06a8d5bf33faf0a4f9a24207ae77b439354350af/src/lightning/fabric/utilities/rank_zero.py#L39-L49
    """
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


def get_local_rank() -> int | None:
    """Get the local rank of the current process."""
    rank_keys = ("LOCAL_RANK", "SLURM_LOCALID", "JSM_NAMESPACE_LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


def get_node_rank() -> int | None:
    """Get the node rank of the current process."""
    rank_keys = ("NODE_RANK", "GROUP_RANK", "SLURM_NODEID")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


def get_master_addr() -> str:
    if SLURMEnvironment.detect():
        nodes = os.environ.get("SLURM_NODELIST")
        nodes = re.sub(
            r"\[(.*?)[,-].*\]", "\\1", nodes
        )  # Take the first node of every node range
        nodes = re.sub(
            r"\[(.*?)\]", "\\1", nodes
        )  # handle special case where node range is single number
        return nodes.split(" ")[0].split(",")[0]
    else:
        return os.environ.get("MASTER_ADDR", "localhost")


def setup_tcp_store() -> TCPStore:
    # Initialize the TCP store for distributed training
    store = TCPStore(
        host_name=get_master_addr(),
        port=12345,
        world_size=get_world_size(),
        timeout=datetime.timedelta(seconds=30),
        is_master=get_global_rank() == 0,
    )
    return store


def main() -> None:
    # Define the model
    model = FNOModel(
        "sfno",
        {
            "n_modes": [24, 24],
            "hidden_channels": 256,
            "n_layers": 5,
            "in_channels": 3,
            "out_channels": 3,
        },
    )

    logdir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_log"
    if SLURMEnvironment.detect():
        # Init TCP store for distributed training
        store = setup_tcp_store()
        store.set("logdir", logdir)
        logdir = store.get("logdir").decode("utf-8")
        print(f"Rank {get_global_rank()} - Logdir: {logdir}")
    else:
        os.environ["LOGDIR"] = logdir
        logdir = os.environ["LOGDIR"]
        print(f"Rank {get_global_rank()} - Logdir: {logdir}")

    # Define the dataset parameters
    dims = (128, 128)
    num_examples = 1000
    batch_size = 16

    # Get the datasets and dataloaders
    train_ds, val_ds = get_trainval_datasets(dims, num_examples)
    train_dl, val_dl = get_trainval_dataloaders(train_ds, val_ds, batch_size)

    # Setup the trainer
    trainer = setup_trainer(devices=2)

    # Train the model
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
