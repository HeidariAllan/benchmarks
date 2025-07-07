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
from moabb.datasets import BNCI2014_001, Cho2017, Lee2019_MI
from moabb.datasets.base import BaseDataset
from moabb.paradigms import MotorImagery
from moabb.paradigms.base import BaseParadigm
from models.SpatialEEGNet import SpatialEEGNet, SpatialFocus
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from sklearn import metrics
from speechbrain.nnet.losses import nll_loss
from speechbrain.processing.signal_processing import mean_std_norm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import logging

# ========================== CONFIG ==========================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESSING_PARAMS = {"fmin": 0.1, "fmax": 50.0, "resample": 128}
CACHE_CONFIG = {"save_epochs": True, "use": True}
PAD_TIME = 640
N_CLASSES = 3
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Using device: {DEVICE}")

# ========================== DATASET CLASS ==========================
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
        ch_positions = torch.from_numpy(
            np.array(list(X.info.get_montage().get_positions()["ch_pos"].values()))
        ).sub_(offset).div_(radius).float().contiguous()
        X = torch.from_numpy(X.get_data()).float()
        X = mean_std_norm(X, dims=(1, 2))
        y = pd.Series(y)
        if self.map_labels:
            y = y.replace(self.map_labels)
        else:
            y = y.replace(self.dataset.event_id) - 1
        y = pd.to_numeric(y, errors="raise", downcast="unsigned")
        y = torch.from_numpy(y.values)
        metadata["session"], _ = metadata["session"].factorize(sort=True)
        metadata["run"], _ = metadata["run"].factorize(sort=True)
        metadata["subject"] += self.subject_offset
        return X, y, metadata, ch_positions

# ========================== HELPERS ==========================
def prepare_LOSO_indices(datasets):
    folds = []
    offset = 0
    for ds in datasets:
        subjects = [ds[i].subject for i in range(len(ds))]
        unique_subjects = sorted(set(subjects))
        for subj in unique_subjects:
            test_idx = [i + offset for i, s in enumerate(subjects) if s == subj]
            train_idx = [i + offset for i, s in enumerate(subjects) if s != subj]
            folds.append({
                "dataset_name": type(ds.dataset).__name__,
                "subject": subj,
                "train_idx": train_idx,
                "test_idx": test_idx
            })
        offset += len(ds)
    return folds

def train_valid_split(dataset, valid_ratio=0.2):
    N = len(dataset)
    N_valid = int(N * valid_ratio)
    return random_split(dataset, [N - N_valid, N_valid])

def compute_class_weights(train_dataset, n_classes):
    labels = [train_dataset[i].y.item() for i in range(len(train_dataset))]
    counts = pd.Series(labels).value_counts().sort_index()
    for cls in range(n_classes):
        if cls not in counts:
            counts.loc[cls] = 0
    counts = counts.sort_index()
    weights = counts / counts.max()
    return torch.from_numpy(weights.values).float().to(DEVICE)

def get_model(**kwargs):
    model = SpatialEEGNet(
        T=PAD_TIME,
        C=kwargs["spatial_focus_projection_dim"],
        cnn_temporal_kernels=kwargs["cnn_temporal_kernels"],
        cnn_temporal_kernelsize=[kwargs["cnn_temporal_kernelsize"], 1],
        cnn_spatial_depth_multiplier=kwargs["cnn_spatial_depth_multiplier"],
        cnn_spatial_max_norm=1,
        cnn_spatial_pool=[4, 1],
        cnn_septemporal_depth_multiplier=1,
        cnn_septemporal_point_kernels=math.ceil(
            (kwargs["cnn_septemporal_point_kernels_ratio_"]/4) *
            (kwargs["cnn_temporal_kernels"] * kwargs["cnn_spatial_depth_multiplier"] * 1) + 1
        ),
        cnn_septemporal_kernelsize=[round(kwargs["cnn_septemporal_kernelsize_"]), 1],
        cnn_septemporal_pool=[kwargs["cnn_septemporal_pool"], 1],
        cnn_pool_type="avg",
        activation_type="elu",
        spatial_focus=SpatialFocus(
            projection_dim=kwargs["spatial_focus_projection_dim"],
            position_dim=3,
            tau=kwargs["spatial_focus_temperature"],
        ),
        dense_max_norm=0.25,
        dropout=kwargs["dropout"],
        dense_n_neurons=N_CLASSES,
    )
    return model.to(DEVICE)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========================== MAIN LOSO MULTI-SEED ==========================
if __name__ == "__main__":
    bnci = TorchMOABBDataset(BNCI2014_001(), MotorImagery(**PREPROCESSING_PARAMS),
                             cache_config=CACHE_CONFIG, map_labels=dict(left_hand=0, right_hand=1, feet=2, tongue=2),
                             pad_time=PAD_TIME, subject_offset=0)
    lee = TorchMOABBDataset(Lee2019_MI(), MotorImagery(**PREPROCESSING_PARAMS),
                            cache_config=CACHE_CONFIG, map_labels=dict(left_hand=0, right_hand=1),
                            pad_time=PAD_TIME, subject_offset=9)
    cho = TorchMOABBDataset(Cho2017(), MotorImagery(**PREPROCESSING_PARAMS),
                            cache_config=CACHE_CONFIG, map_labels=dict(left_hand=0, right_hand=1),
                            pad_time=PAD_TIME, subject_offset=63)

    dataset_list = [bnci, lee, cho]
    folds = prepare_LOSO_indices(dataset_list)
    concat = ConcatDataset(dataset_list)
    results = []

    for seed in [42, 1234, 5678]:
        set_seed(seed)
        logger.info(f"===> Seed {seed} LOSO start...")
        for fold in folds:
            logger.info(f"Dataset {fold['dataset_name']} Subj {fold['subject']}")
            train_ds = Subset(concat, fold['train_idx'])
            test_ds = Subset(concat, fold['test_idx'])  # ✅ REAL held-out subject!
            train_ds, valid_ds = train_valid_split(train_ds)
            class_weights = compute_class_weights(train_ds, N_CLASSES)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            model = get_model(
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
            swa_scheduler = SWALR(optimizer, anneal_strategy="linear",
                                   anneal_epochs=76, swa_lr=0.05)
            swa_model = AveragedModel(model)
            swa_start = 76

            model.train()
            for epoch in range(1, 101):
                running_loss = 0.0
                optimizer.zero_grad()
                for idx, batch in enumerate(train_loader):
                    output = model(batch)
                    loss = nll_loss(output, batch.y, weight=class_weights)
                    loss.backward()
                    running_loss += loss.item()
                    if idx % GRADIENT_ACCUMULATION == GRADIENT_ACCUMULATION - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                if epoch >= swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()

            torch.optim.swa_utils.update_bn(train_loader, swa_model)

            # ✅ Final test on held-out subject
            swa_model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in test_loader:
                    output = swa_model(batch)
                    y_pred.extend(torch.argmax(output, dim=-1).cpu().numpy())
                    y_true.extend(batch.y.cpu().numpy())
            acc = metrics.balanced_accuracy_score(y_true, y_pred)
            f1 = metrics.f1_score(y_true, y_pred, average="macro")
            logger.info(f"Fold: {fold['dataset_name']} Subj {fold['subject']} Seed {seed} TEST Acc={acc:.4f} F1={f1:.4f}")
            results.append({"dataset": fold['dataset_name'], "subject": fold['subject'],
                            "seed": seed, "acc": acc, "f1": f1})

    df = pd.DataFrame(results)
    print(df.groupby(["dataset"]).agg({"acc": ["mean"], "f1": ["mean"]}))
    df.to_csv("loso_all_seeds.csv", index=False)
    print(df)
