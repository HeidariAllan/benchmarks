import logging
import random
import math
from typing import Sequence, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from hyperpyyaml import load_hyperpyyaml

from moabb.datasets import BNCI2014_001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
from speechbrain.processing.signal_processing import mean_std_norm
from sklearn.metrics import classification_report, balanced_accuracy_score

# =====================
# Logging Setup
# =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================
# Dataset Wrapper
# =====================
class TorchMOABBDataset(Dataset):
    def __init__(
        self,
        dataset,
        paradigm,
        subjects: Optional[Sequence[int]] = None,
        map_labels: Optional[Dict[str, int]] = None,
        cache_config: Optional[Dict] = None,
        head_sphere: tuple = ((0, 0, 0.04), 0.09),
        pad_time: Optional[int] = None,
    ):
        self.dataset = dataset
        self.paradigm = paradigm
        self.subjects = subjects or dataset.subject_list
        self.map_labels = map_labels
        self.cache_config = cache_config
        self.head_sphere = head_sphere
        self.pad_time = pad_time

        self._load_data()

    def _load_data(self):
        X, y, metadata = self.paradigm.get_data(
            self.dataset,
            subjects=self.subjects,
            return_epochs=True,
            cache_config=self.cache_config,
        )

        offset = torch.tensor(self.head_sphere[0])
        radius = self.head_sphere[1]
        ch_positions = (
            torch.from_numpy(
                np.array(
                    list(
                        X.info.get_montage().get_positions()["ch_pos"].values()
                    )
                )
            )
            .sub_(offset)
            .div_(radius)
            .float()
        )

        X = torch.from_numpy(X.get_data()).float()
        X = mean_std_norm(X, dims=(1, 2))

        y = pd.Series(y)
        label_map = self.map_labels or self.dataset.event_id
        y = y.replace(label_map).astype(int)
        y = torch.from_numpy(y.values)

        metadata["session"], _ = metadata["session"].factorize()
        metadata["run"], _ = metadata["run"].factorize()

        self.X, self.y, self.metadata, self.ch_positions = X, y, metadata, ch_positions

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.pad_time:
            x = torch.nn.functional.pad(x, (0, self.pad_time - x.shape[1], 0, 0))
        meta = self.metadata.iloc[idx]
        return Data(
            x=x.to(self.X.device),
            y=self.y[idx].to(self.y.device),
            pos=self.ch_positions.to(self.X.device),
            subject=meta["subject"],
            session=meta["session"],
            run=meta["run"],
        )


# =====================
# Prepare LOSO Split
# =====================
def prepare_datasets_loso(hparams) -> (ConcatDataset, ConcatDataset):
    paradigm = hparams["paradigm"]
    dataset_classes = hparams["datasets"]
    label_maps = hparams["label_maps"]
    cache_config = hparams["cache_config"]
    pad_time = hparams["pad_time"]

    train_sets = []
    valid_sets = []

    for ds_cls in dataset_classes:
        dataset = ds_cls()
        name = dataset.__class__.__name__
        map_labels = label_maps.get(name, {})
        all_subjects = dataset.subject_list
        left_out = all_subjects[0]

        train_sets.append(
            TorchMOABBDataset(
                dataset,
                paradigm,
                subjects=[s for s in all_subjects if s != left_out],
                map_labels=map_labels,
                cache_config=cache_config,
                pad_time=pad_time,
            )
        )
        valid_sets.append(
            TorchMOABBDataset(
                dataset,
                paradigm,
                subjects=[left_out],
                map_labels=map_labels,
                cache_config=cache_config,
                pad_time=pad_time,
            )
        )
        logger.info(f"{name}: left out subject {left_out}")

    return ConcatDataset(train_sets), ConcatDataset(valid_sets)


# =====================
# Class Weights
# =====================
def compute_class_weights(dataset, n_classes, device):
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    counts = pd.Series(labels).value_counts().reindex(range(n_classes), fill_value=0)
    weights = counts.max() / counts
    return torch.tensor(weights.values, dtype=torch.float32).to(device)


# =====================
# Training + Evaluation
# =====================
def train(model, optimizer, loss_fn, train_loader, device, num_epochs):
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"[Epoch {epoch}] Loss: {total_loss / len(train_loader):.4f}")


def evaluate(model, valid_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = torch.argmax(out, dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"\nBalanced Accuracy: {acc:.4f}\n{report}")


# =====================
# Main Execution
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(hyperyaml_path="hyperparams.yaml"):
    with open(hyperyaml_path) as f:
        hparams = load_hyperpyyaml(f)

    set_seed(hparams["seed"])
    device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset, valid_dataset = prepare_datasets_loso(hparams)
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=hparams["batch_size"])

    model = hparams["model"].to(device)
    optimizer = hparams["optimizer"]
    loss_fn = hparams["loss"]
    class_weights = compute_class_weights(train_dataset, hparams["n_classes"], device)

    def weighted_loss(output, target):
        return loss_fn(output, target, weight=class_weights)

    logger.info("Training model...")
    train(model, optimizer, weighted_loss, train_loader, device, hparams["number_of_epochs"])

    logger.info("Evaluating model...")
    evaluate(model, valid_loader, device)


if __name__ == "__main__":
    main()
