import random
import logging
from typing import Optional, Sequence, Dict, Tuple
import pathlib as pl

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from moabb.datasets import BNCI2014_001, Cho2017, PhysionetMI
from moabb.datasets.base import BaseDataset
from moabb.paradigms import MotorImagery
from moabb.paradigms.base import BaseParadigm
from speechbrain.processing.signal_processing import mean_std_norm
from speechbrain.nnet.losses import nll_loss
from models.SpatialEEGNet import SpatialEEGNet, SpatialFocus

# =============== Config ===============
SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESSING_PARAMS = {"fmin": 0.1, "fmax": 50.0, "resample": 128}
CACHE_CONFIG = {"save_epochs": True, "use": True}
PAD_TIME = 513
N_CLASSES = 3
BATCH_SIZE = 8
NUM_EPOCHS = 20
WORKING_DIR = "results/loso_experiment"
pl.Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)

# =============== Logging ===============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============== Dataset Wrapper ===============
class TorchMOABBDataset(Dataset):
    def __init__(
        self,
        dataset: BaseDataset,
        paradigm: BaseParadigm,
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
        y = y.replace(self.map_labels or self.dataset.event_id) - 1
        y = torch.from_numpy(pd.to_numeric(y, errors="raise").values)

        metadata["session"], _ = metadata["session"].factorize(sort=True)
        metadata["run"], _ = metadata["run"].factorize(sort=True)

        self.X, self.y, self.metadata, self.ch_positions = X, y, metadata, ch_positions

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if self.pad_time:
            x = torch.nn.functional.pad(x, (0, self.pad_time - x.shape[1], 0, 0))

        meta = self.metadata.iloc[index]
        return Data(
            x=x.to(DEVICE),
            y=self.y[index].to(DEVICE),
            pos=self.ch_positions.to(DEVICE),
            subject=meta["subject"],
            session=meta["session"],
            run=meta["run"],
        )

# =============== Prepare LOSO Datasets ===============
def prepare_datasets_with_loso() -> Tuple[ConcatDataset, ConcatDataset]:
    paradigm = MotorImagery(**PREPROCESSING_PARAMS)

    datasets = [
        BNCI2014_001(),
        Cho2017(),
        PhysionetMI()
    ]

    label_maps = [
        dict(left_hand=0, right_hand=1, feet=2, tongue=2),
        dict(left_hand=0, right_hand=1),
        dict(left_hand=0, right_hand=1)
    ]

    train_sets, valid_sets = [], []

    for dataset, label_map in zip(datasets, label_maps):
        all_subjects = dataset.subject_list
        left_out = all_subjects[0]  # LOSO

        train_sets.append(
            TorchMOABBDataset(dataset, paradigm, subjects=[s for s in all_subjects if s != left_out],
                              map_labels=label_map, cache_config=CACHE_CONFIG, pad_time=PAD_TIME)
        )
        valid_sets.append(
            TorchMOABBDataset(dataset, paradigm, subjects=[left_out],
                              map_labels=label_map, cache_config=CACHE_CONFIG, pad_time=PAD_TIME)
        )
        logger.info(f"{dataset.__class__.__name__}: Left out subject {left_out} for validation.")

    return ConcatDataset(train_sets), ConcatDataset(valid_sets)

# =============== Model ===============
def get_model() -> SpatialEEGNet:
    return SpatialEEGNet(
        T=PAD_TIME,
        C=22,
        cnn_temporal_kernels=40,
        cnn_temporal_kernelsize=[30, 1],
        cnn_spatial_depth_multiplier=2,
        cnn_spatial_max_norm=1,
        cnn_spatial_pool=[4, 1],
        cnn_septemporal_depth_multiplier=1,
        cnn_septemporal_point_kernels=64,
        cnn_septemporal_kernelsize=[15, 1],
        cnn_septemporal_pool=[2, 1],
        cnn_pool_type="avg",
        activation_type="elu",
        spatial_focus=SpatialFocus(projection_dim=22, position_dim=3, tau=0.5),
        dense_max_norm=0.25,
        dropout=0.3,
        dense_n_neurons=N_CLASSES,
    ).to(DEVICE)

# =============== Class Weights ===============
def compute_class_weights(dataset: Dataset, n_classes: int) -> torch.Tensor:
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    counts = pd.Series(labels).value_counts().reindex(range(n_classes), fill_value=0)
    weights = counts.max() / counts
    return torch.tensor(weights.values, dtype=torch.float32).to(DEVICE)

# =============== Training ===============
def train(train_loader, model, optimizer, class_weights):
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        for batch in train_loader:
            output = model(batch)
            loss = nll_loss(output, batch.y, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

# =============== Evaluation ===============
def evaluate(model, valid_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in valid_loader:
            output = model(batch)
            pred = torch.argmax(output, dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    from sklearn.metrics import balanced_accuracy_score, classification_report
    acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"Balanced Accuracy: {acc:.4f}")
    logger.info(f"Classification Report:\n{report}")

# =============== Main ===============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(SEED)

    logger.info("Preparing LOSO datasets...")
    train_dataset, valid_dataset = prepare_datasets_with_loso()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    logger.info("Building model...")
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    class_weights = compute_class_weights(train_dataset, N_CLASSES)

    logger.info("Starting training...")
    train(train_loader, model, optimizer, class_weights)

    logger.info("Evaluating model...")
    evaluate(model, valid_loader)
