import json
import random

import math
import pathlib as pl
import warnings
from contextlib import redirect_stdout
from functools import cached_property
from pprint import pformat
from typing import Any, Optional, Sequence, Dict, List

import numpy as np
import pandas as pd
import torch
from moabb.datasets import BNCI2014_001, Cho2017, BNCI2014_004, Lee2019_MI
from moabb.datasets.base import BaseDataset
from moabb.paradigms import MotorImagery
from moabb.paradigms.base import BaseParadigm
from models.SpatialEEGNet import SpatialEEGNet, SpatialFocus
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from sklearn import metrics
from speechbrain.nnet.losses import nll_loss
from speechbrain.processing.signal_processing import mean_std_norm
from torch.utils.data import ConcatDataset, Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import logging
from torch.utils.tensorboard import SummaryWriter

# ==========================
# Configuration and Constants
# ==========================
SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESSING_PARAMS = {
    "fmin": 0.1,
    "fmax": 50.0,
    "resample": 128,
}

CACHE_CONFIG = {"save_epochs": True, "use": True}

PAD_TIME = 640
N_CLASSES = 3
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4  # Effective Batchsize = 32
EXPERIMENT_NAME = "spatial-eegnet-beetl-MI"
WORKING_DIR = f"results/hopt/{EXPERIMENT_NAME}"
pl.Path(WORKING_DIR).mkdir(exist_ok=True, parents=True)

# ==========================
# Logging Configuration
# ==========================

# Configure logging to output to both console and a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{WORKING_DIR}/training.log"),
        logging.StreamHandler(),
    ],
)
logging.getLogger("moabb").setLevel(logging.ERROR)
logging.getLogger("mne").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.info(f"Using device: {DEVICE}")

# ==========================
# Dataset Preparation
# ==========================


class TorchMOABBDataset(Dataset):
    def __init__(
        self,
        dataset: BaseDataset,
        paradigm: BaseParadigm,
        subjects: Optional[Sequence[int]] = None,
        map_labels: Optional[Dict[str, int]] = None,
        cache_config: Optional[Dict] = None,
        head_sphere: tuple = ((0, 0, 0.04), 0.09),  # offset, radius
        pad_time: Optional[int] = None,
        subject_offset: int = 0, 
    ):
        self.dataset = dataset
        self.paradigm = paradigm
        self.subjects = subjects or dataset.subject_list
        self.map_labels = map_labels
        self.cache_config = cache_config
        self.head_sphere = head_sphere
        self.pad_time = pad_time
        self.subject_offset = subject_offset

    def __len__(self):
        return len(self._data[0])

    def __getitem__(self, index):
        X, y, metadata, ch_positions = self._data
        meta = metadata.iloc[index]

        x = X[index]
        if self.pad_time:
            assert self.pad_time >= x.shape[1], "Expected T to be less than pad_time"
            x = torch.nn.functional.pad(x, (0, self.pad_time - x.shape[1], 0, 0))

        return Data(
            x=x.to(DEVICE),
            y=y[index].to(DEVICE),
            pos=ch_positions.to(DEVICE),
            subject=meta["subject"],
            session=meta["session"],
            run=meta["run"],
        )

    @cached_property
    def _data(self):
        from io import StringIO

        with warnings.catch_warnings(), redirect_stdout(StringIO()):
            warnings.simplefilter("ignore")
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
                np.array(list(X.info.get_montage().get_positions()["ch_pos"].values()))
            )
            .sub_(offset)
            .div_(radius)
            .float()
            .contiguous()
        )

        X = torch.from_numpy(X.get_data()).float()
        X = mean_std_norm(X, dims=(1, 2))

        y = pd.Series(y)
        if self.map_labels:
            y = y.replace(self.map_labels)
        else:
            y = y.replace(self.dataset.event_id) - 1
        y = pd.to_numeric(y, errors="raise", downcast="unsigned")
        y = torch.from_numpy(y.values)

        metadata["session"], self._session_names = metadata["session"].factorize(sort=True)
        metadata["run"], self._run_names = metadata["run"].factorize(sort=True)
        metadata["subject"] += self.subject_offset  # apply unique offset here

        return X, y, metadata, ch_positions


def prepare_datasets() -> ConcatDataset:
    bnci = TorchMOABBDataset(
        dataset=BNCI2014_001(),
        paradigm=MotorImagery(**PREPROCESSING_PARAMS),
        cache_config=CACHE_CONFIG,
        map_labels=dict(left_hand=0, right_hand=1, feet=2, tongue=2),
        pad_time=PAD_TIME,
        subject_offset=0,
    )

    lee = TorchMOABBDataset(
      dataset=Lee2019_MI(),
      paradigm=MotorImagery(**PREPROCESSING_PARAMS),
      cache_config=CACHE_CONFIG,
      map_labels=dict(left_hand=0, right_hand=1),
      pad_time=PAD_TIME,
      subject_offset=9,
    )

    cho = TorchMOABBDataset(
        dataset=Cho2017(),
        paradigm=MotorImagery(**PREPROCESSING_PARAMS),
        cache_config=CACHE_CONFIG,
        map_labels=dict(left_hand=0, right_hand=1),
        pad_time=PAD_TIME,
        subject_offset=63,
    )
    #new
    # bnci004 = TorchMOABBDataset(
    #   dataset=BNCI2014_004(),
    #   paradigm=MotorImagery(**PREPROCESSING_PARAMS),
    #   cache_config=CACHE_CONFIG,
    #   map_labels=dict(left_hand=0, right_hand=1),
    #   pad_time=PAD_TIME,
    #   subject_offset=0,
    # )


    # return ConcatDataset([bnci, cho])

    # datasets = [bnci, cho, lee, bnci004]
    datasets = [bnci, lee, cho]
    test_index = []
    train_index = []

    # Step 1: Precompute subjects for each dataset
    subjects_per_ds = []
    for ds in datasets:
        subjects = []
        for i in range(len(ds)):
            # This calls __getitem__ once per sample to get metadata only
            subjects.append(ds[i].subject)
        subjects_per_ds.append(subjects)

    # Step 2: LOSO split with global offset
    offset = 0
    for ds, subjects in zip(datasets, subjects_per_ds):
        loso = [i for i, subj in enumerate(subjects) if subj == 1]
        train = [i for i, subj in enumerate(subjects) if subj != 1]

        test_index.extend([i + offset for i in loso])
        train_index.extend([i + offset for i in train])

        offset += len(ds)

    #Combine and create subsets
    concat = ConcatDataset(datasets)
    train_ds = Subset(concat, train_index)
    test_ds = Subset(concat, test_index)

    return train_ds, test_ds


def train_valid_split(dataset: ConcatDataset, valid_ratio: float = 0.2):
    N = len(dataset)
    N_valid = int(N * valid_ratio)
    return random_split(dataset, [N - N_valid, N_valid])


def compute_class_weights(train_dataset: Dataset, n_classes: int) -> torch.Tensor:
    train_labels = [train_dataset[i].y.item() for i in range(len(train_dataset))]
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    for cls in range(n_classes):
        if cls not in class_counts:
            class_counts.loc[cls] = 0
    class_counts = class_counts.sort_index()
    class_weights = class_counts / class_counts.max()
    class_weights = torch.from_numpy(class_weights.values).float().to(DEVICE)
    return class_weights


def get_model(
    T: int,
    n_classes: int,
    spatial_focus_projection_dim: int,
    spatial_focus_temperature: float,
    cnn_temporal_kernels: int,
    cnn_temporal_kernelsize: int,
    cnn_spatial_depth_multiplier: int,
    cnn_septemporal_point_kernels_ratio_: float,
    cnn_septemporal_kernelsize_: int,
    cnn_septemporal_pool: int,
    dropout: float,
) -> SpatialEEGNet:
    activation_type = "elu"
    cnn_spatial_max_norm = 1
    cnn_spatial_pool = 4
    cnn_septemporal_depth_multiplier = 1

    cnn_septemporal_point_kernels_ratio = cnn_septemporal_point_kernels_ratio_ / 4
    cnn_septemporal_point_kernels_ = (
        cnn_temporal_kernels * cnn_spatial_depth_multiplier * cnn_septemporal_depth_multiplier
    )
    cnn_septemporal_point_kernels = math.ceil(
        cnn_septemporal_point_kernels_ratio * cnn_septemporal_point_kernels_ + 1
    )
    max_cnn_spatial_pool = 4
    cnn_septemporal_kernelsize = round(
        cnn_septemporal_kernelsize_ * max_cnn_spatial_pool / cnn_spatial_pool
    )
    cnn_pool_type = "avg"
    dense_max_norm = 0.25

    model = SpatialEEGNet(
        T=T,
        C=spatial_focus_projection_dim,
        cnn_temporal_kernels=cnn_temporal_kernels,
        cnn_temporal_kernelsize=[cnn_temporal_kernelsize, 1],
        cnn_spatial_depth_multiplier=cnn_spatial_depth_multiplier,
        cnn_spatial_max_norm=cnn_spatial_max_norm,
        cnn_spatial_pool=[cnn_spatial_pool, 1],
        cnn_septemporal_depth_multiplier=cnn_septemporal_depth_multiplier,
        cnn_septemporal_point_kernels=cnn_septemporal_point_kernels,
        cnn_septemporal_kernelsize=[cnn_septemporal_kernelsize, 1],
        cnn_septemporal_pool=[cnn_septemporal_pool, 1],
        cnn_pool_type=cnn_pool_type,
        activation_type=activation_type,
        spatial_focus=SpatialFocus(
            projection_dim=spatial_focus_projection_dim,
            position_dim=3,
            tau=spatial_focus_temperature,
        ),
        dense_max_norm=dense_max_norm,
        dropout=dropout,
        dense_n_neurons=n_classes,
    )
    return model.to(DEVICE)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(seed: int, dataset=None):
    logger.info(f"Starting run for seed: {seed}")
    set_seed(seed)

    # === Prepare data ===
    if dataset is None:
        dataset, test_set = prepare_datasets()
    else:
        _, test_set = prepare_datasets()  # Or reuse your test_set if you store it
    
    train_dataset, valid_dataset = train_valid_split(dataset)
    class_weights = compute_class_weights(train_dataset, N_CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # === Init model ===
    model = get_model(
        T=PAD_TIME,
        n_classes=N_CLASSES,
        spatial_focus_projection_dim=32,
        spatial_focus_temperature=0.5,
        cnn_temporal_kernels=40,
        cnn_temporal_kernelsize=40,
        cnn_spatial_depth_multiplier=2,
        cnn_septemporal_point_kernels_ratio_=4,
        cnn_septemporal_kernelsize_=15,
        cnn_septemporal_pool=4,
        dropout=0.3,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    swa_scheduler = SWALR(
        optimizer,
        anneal_strategy="linear",
        anneal_epochs=1 + int(0.75 * 100),
        swa_lr=0.05,
    )
    swa_model = AveragedModel(model)
    swa_start = 1 + int(0.75 * 100)
    logger.info(f"SWA will start after epoch {swa_start}")

    model.train()
    for epoch in range(1, 101):
        running_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for idx, batch in enumerate(train_loader):
            output = model(batch)
            loss = nll_loss(output, batch.y, weight=class_weights)
            loss.backward()

            running_loss += loss.item()
            num_batches += 1

            if idx % GRADIENT_ACCUMULATION == GRADIENT_ACCUMULATION - 1:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = running_loss / num_batches
        logger.info(f"Seed {seed} - Epoch {epoch} - Loss: {avg_loss:.4f}")

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

    # === Evaluate ===
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    swa_model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in valid_loader:
            output = swa_model(batch)
            y_pred = torch.argmax(output, dim=-1).cpu().numpy()
            y_true = batch.y.cpu().numpy()
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc = metrics.balanced_accuracy_score(y_true_all, y_pred_all)
    f1 = metrics.f1_score(y_true_all, y_pred_all, average="macro")
    cm = metrics.confusion_matrix(y_true_all, y_pred_all).tolist()

    logger.info(f"Seed {seed} - Balanced Accuracy: {acc:.4f}")
    logger.info(f"Seed {seed} - F1 Score: {f1:.4f}")
    logger.info(f"Seed {seed} - Confusion Matrix: {cm}")

    result = {
        "seed": seed,
        "acc": acc,
        "f1": f1,
        "cm": cm
    }

    return result

if __name__ == "__main__":
    all_results = []
    for seed in [42, 1234, 5678]:
        result = run_experiment(seed)
        all_results.append(result)

    avg_acc = np.mean([r["acc"] for r in all_results])
    avg_f1 = np.mean([r["f1"] for r in all_results])

    print("=== Final Results ===")
    for res in all_results:
        print(f"Seed {res['seed']}: Acc={res['acc']:.4f}, F1={res['f1']:.4f}")
    print(f"Mean Acc: {avg_acc:.4f}, Mean F1: {avg_f1:.4f}")
