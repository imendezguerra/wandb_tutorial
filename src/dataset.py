"""Dataset generator."""
import numpy as np
from typing import Dict
from pathlib import Path
from loguru import logger
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from src.plots import plot_split_all, plot_class_counts
from src.io import yaml_save
from torch.utils.data import Dataset, DataLoader
import torch

def get_class_counts(y):
    labels, counts = np.unique(y, return_counts=True)
    return dict(zip(labels, counts))

def generate_dataset(
    config: Dict,
    output_path: Path,
    ):

    # Save config
    Path(output_path).mkdir(parents=True, exist_ok=True)
    yaml_save(config, output_path / 'config.yml')

    # Build dataset based on config
    X, y = make_blobs(
        n_samples=config['n_samples']*config['n_classes'],
        centers=config['n_classes'],
        cluster_std=config['cluster_std'],
        n_features=config['n_features'],
        random_state=config['seed'],
    )
    logger.info('Blobs generated')

    # Separate into train/val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['seed'],
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=config['val_size'],
        random_state=config['seed'],
        stratify=y_trainval
    )
    logger.info('Data separated into:')
    for X_log, y_log, split in (
        (X_train, y_train, 'Train split'),
        (X_val, y_val, 'Validation split'),
        (X_test, y_test, 'Test split'),
    ):
        logger.info(split)
        logger.info("-" * 15)
        logger.info(f"X: {X_log.shape}")
        logger.info(f"y: {y_log.shape}")

    # Save data
    np.savez(output_path/"data_train.npz", X=X_train, y=y_train)
    np.savez(output_path/"data_val.npz", X=X_val, y=y_val)
    np.savez(output_path/"data_test.npz", X=X_test, y=y_test)
    logger.info(f"Saved dataset to {output_path}/*.npz")

    # Plot dataset
    fig = plot_split_all(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    fig.savefig(output_path/"data_split.png")

    # Plot class counts
    train_counts = get_class_counts(y_train)
    val_counts   = get_class_counts(y_val)
    test_counts  = get_class_counts(y_test)
    fig = plot_class_counts(
        train_counts, val_counts, test_counts, config['n_classes']
    )
    fig.savefig(output_path/"data_class_split.png")

class BlobDataset(Dataset):
    def __init__(self, npz_path, device: str = "cpu"):
        data = np.load(npz_path)
        self.device = torch.device(device)
        self.X = torch.from_numpy(data["X"]).to(dtype=torch.float32, device=self.device)
        self.y = torch.from_numpy(data["y"]).to(dtype=torch.int64, device=self.device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def get_dataloaders_from_artifact(artifact_dir:str, batch_size:int, device:str="cpu"):

    train_path = Path(artifact_dir) / "data_train.npz"
    val_path = Path(artifact_dir) / "data_val.npz"
    test_path = Path(artifact_dir) / "data_test.npz"

    train_ds = BlobDataset(train_path, device=device)
    val_ds = BlobDataset(val_path, device=device)
    test_ds = BlobDataset(test_path, device=device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

