#!/usr/bin/python
"""
Cross-dataset training with leave-one-subject-out (LOSO) and SWA support.
Uses standalone training loop instead of SpeechBrain Brain class.
"""

import os
import sys

# CRITICAL: Set MNE_DATA before importing MOABB to use local datasets
os.environ['MNE_DATA'] = os.environ.get('SLURM_TMPDIR', '/tmp') + '/mne_data'

import logging
import pickle
import numpy as np
import torch
import yaml
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset, Dataset, Subset
from functools import cached_property
import warnings
from contextlib import redirect_stdout
import pandas as pd
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn import metrics

from moabb.datasets import BNCI2014_001, Cho2017, Lee2019_MI
from moabb.paradigms import MotorImagery

# CRITICAL: Patch MOABB download function to use local files only
import moabb.datasets.download as moabb_dl
from urllib.parse import urlparse

def local_first_data_dl(url, sign, path=None, force_update=False, verbose=None):
    """Check if file exists locally before attempting download"""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)

    if path is None:
        path = os.environ.get('MNE_DATA', os.path.expanduser('~/mne_data'))

    # Construct local path based on dataset type
    if 'bnci' in url.lower():
        local_path = os.path.join(path, 'MNE-bnci-data', parsed.path.lstrip('/'))
    elif '100542' in url:  # Lee2019_MI specific dataset ID
        local_path = os.path.join(path, 'MNE-lee2019-mi-data', parsed.path.lstrip('/'))
    elif 'gigadb' in url.lower():  # Cho2017
        local_path = os.path.join(path, 'MNE-gigadb-data', parsed.path.lstrip('/'))
    else:
        local_path = os.path.join(path, 'MNE-lee2019-mi-data', parsed.path.lstrip('/'))

    if os.path.exists(local_path) and not force_update:
        return local_path
    else:
        raise FileNotFoundError(f"File not found locally and downloads disabled: {local_path}")

# Apply the patch
moabb_dl.data_dl = local_first_data_dl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Device setup - always GPU
DEVICE = torch.device("cuda")


# ------------------------------
# Seed setting for reproducibility
# ------------------------------
def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")


# ------------------------------
# TorchMOABBDataset wrapper
# ------------------------------
class TorchMOABBDataset(Dataset):
    def __init__(
        self,
        dataset,
        paradigm,
        subjects=None,
        map_labels=None,
        cache_config=None,
        head_sphere=((0, 0, 0.04), 0.09),
        pad_time=None,
    ):
        self.dataset = dataset
        self.paradigm = paradigm
        self.subjects = subjects or dataset.subject_list
        self.map_labels = map_labels
        self.cache_config = cache_config
        self.head_sphere = head_sphere
        self.pad_time = pad_time

    def __len__(self):
        return len(self._data[0])

    def __getitem__(self, index):
        X, y, metadata, ch_positions = self._data
        meta = metadata.iloc[index]

        x = X[index]
        if self.pad_time:
            assert self.pad_time >= x.shape[1], "Expected T <= pad_time"
            x = torch.nn.functional.pad(
                x, (0, self.pad_time - x.shape[1], 0, 0)
            )

        return Data(
            x=x,
            y=y[index],
            pos=ch_positions,
            subject=meta["subject"],
            session=meta["session"],
            run=meta["run"],
        )

    @cached_property
    def _data(self):
        from speechbrain.processing.signal_processing import mean_std_norm

        with warnings.catch_warnings(), redirect_stdout(None):
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
                np.array(
                    list(X.info.get_montage().get_positions()["ch_pos"].values())
                )
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

        metadata["session"], _ = metadata["session"].factorize(sort=True)
        metadata["run"], _ = metadata["run"].factorize(sort=True)

        return X, y, metadata, ch_positions


# ------------------------------
# Data loader preparation
# ------------------------------
def prepare_dataloaders(datasets, batch_size):
    """
    Create train and test data loaders.
    No validation set - pure LOSO paradigm.
    """
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader


def compute_class_weights(train_dataset, n_classes):
    """
    Compute class weights for balanced loss.
    """
    # Collect labels from the dataset
    if isinstance(train_dataset, Subset):
        # Collect labels only from the subset indices
        train_labels = [
            train_dataset.dataset[idx].y.item() for idx in train_dataset.indices
        ]
    else:
        # For regular datasets, collect all labels
        train_labels = [
            train_dataset[i].y.item() for i in range(len(train_dataset))
        ]
    
    # Compute class counts using pandas
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    
    # Ensure all classes are present
    if len(class_counts) < n_classes:
        for cls in range(n_classes):
            if cls not in class_counts:
                class_counts.loc[cls] = 0
    class_counts = class_counts.sort_index()
    
    # Compute weights: normalized frequency
    class_weights = class_counts / class_counts.max()
    class_weights = torch.from_numpy(class_weights.values).float().to(DEVICE)
    
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    return class_weights


# ------------------------------
# Training function with SWA
# ------------------------------
def run_model(hparams, run_opts, datasets):
    """
    Standalone training loop with SWA support for LOSO.
    SWA is always enabled.
    """
    # Set random seed for reproducibility
    seed = hparams.get("seed", 1234)
    set_seed(seed)

    logger.info(f"Using device: {DEVICE}")

    # Prepare data loaders
    train_loader, test_loader = prepare_dataloaders(
        datasets, hparams["batch_size"]
    )

    # Compute class weights
    class_weights = compute_class_weights(
        datasets["train"], hparams["n_classes"]
    )

    # Initialize model
    model = hparams["model"].to(DEVICE)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Enable cuDNN autotuner for better performance
    torch.backends.cudnn.benchmark = True

    # Setup optimizer and scheduler
    optimizer = hparams["optimizer"](model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams["number_of_epochs"]
    )

    # SWA setup (always enabled)
    swa_start_ratio = hparams.get("swa_start_ratio", 0.75)
    swa_start = 1 + int(swa_start_ratio * hparams["number_of_epochs"])
    swa_lr = hparams.get("swa_lr", 0.05)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(
        optimizer,
        anneal_strategy="linear",
        anneal_epochs=swa_start,
        swa_lr=swa_lr,
    )
    logger.info(f"SWA will start at epoch {swa_start}/{hparams['number_of_epochs']}")
    logger.info(f"SWA learning rate: {swa_lr}")

    # Loss function
    loss_fn = hparams["loss"]

    # Training loop
    gradient_accumulation = hparams.get("gradient_accumulation", 4)
    logger.info(f"Starting training with gradient accumulation steps: {gradient_accumulation}")

    for epoch in range(1, hparams["number_of_epochs"] + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()
        for idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = batch.to(DEVICE)

            # Forward pass
            output = model(batch)
            loss = loss_fn(output, batch.y, weight=class_weights)

            # Backward pass
            loss.backward()

            running_loss += loss.item()
            num_batches += 1

            # Update weights every gradient_accumulation batches
            if idx % gradient_accumulation == gradient_accumulation - 1:
                optimizer.step()
                optimizer.zero_grad()

        # Calculate average loss
        avg_loss = running_loss / num_batches

        # Update schedulers and SWA model
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{hparams['number_of_epochs']} - Loss: {avg_loss:.4f} - SWA active")
        else:
            scheduler.step()
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{hparams['number_of_epochs']} - Loss: {avg_loss:.4f}")

    # Update SWA batch norm statistics
    logger.info("Updating SWA batch normalization statistics...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)

    logger.info("Using SWA model for evaluation")

    # Evaluate on test set (left-out subject)
    logger.info("Evaluating on test set (left-out subject)...")
    swa_model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            output = swa_model(batch)
            y_pred = torch.argmax(output, dim=-1).cpu().numpy()
            y_true = batch.y.cpu().numpy()

            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

    # Compute metrics
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    test_metrics = {}
    for metric_name, metric_func in hparams["metrics"].items():
        if metric_name == "cm":
            test_metrics[metric_name] = metric_func(y_true_all, y_pred_all).tolist()
        else:
            test_metrics[metric_name] = metric_func(y_true=y_true_all, y_pred=y_pred_all)

    logger.info(f"Test metrics: {test_metrics}")

    # Get dataset name for this subject
    target_subject = hparams["target_subject_idx"]
    dataset_name = hparams["subject_to_dataset"].get(target_subject, "Unknown")
    logger.info(f"Left-out subject {target_subject} is from dataset: {dataset_name}")

    # Add dataset info to metrics
    test_metrics["subject_id"] = target_subject
    test_metrics["dataset_name"] = dataset_name

    # Save results
    save_dir = os.path.join(hparams["exp_dir"], "save")
    os.makedirs(save_dir, exist_ok=True)

    # Save SWA model only
    swa_model_path = os.path.join(save_dir, "swa_model.ckpt")
    torch.save(
        swa_model.module.state_dict() if hasattr(swa_model, "module") else swa_model.state_dict(),
        swa_model_path
    )
    logger.info(f"Saved SWA model to {swa_model_path}")

    # Save metrics
    test_metrics_path = os.path.join(save_dir, "test_metrics.pkl")
    with open(test_metrics_path, "wb") as f:
        pickle.dump(test_metrics, f)

    logger.info("Training completed successfully!")
    return test_metrics


# ------------------------------
# Dataset preparation
# ------------------------------
def prepare_merged_dataset(hparams):
    """
    Prepare and merge BNCI, Cho, and Lee datasets.
    """
    logger.info("Preparing merged dataset (BNCI+Cho+Lee)...")

    pad_time = 640
    merged = []
    subject_offset = 1
    subject_to_dataset = {}

    dataset_root = hparams["data_folder"]
    logger.info(f"Using dataset root: {dataset_root}")

    bnci = BNCI2014_001()
    cho = Cho2017()
    lee = Lee2019_MI()

    dataset_info = [
        (bnci, "BNCI2014_001", dict(left_hand=0, right_hand=1, feet=2, tongue=2)),
        (cho, "Cho2017", dict(left_hand=0, right_hand=1)),
        (lee, "Lee2019_MI", dict(left_hand=0, right_hand=1))
    ]

    for ds, dataset_name, label_map in dataset_info:
        paradigm = MotorImagery(
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            resample=hparams["sample_rate"],
        )

        torch_ds = TorchMOABBDataset(
            dataset=ds,
            paradigm=paradigm,
            cache_config=hparams.get("cache_config"),
            pad_time=pad_time,
            map_labels=label_map,
        )

        original_subjects = sorted(torch_ds._data[2]["subject"].unique())
        unique_subjects = len(original_subjects)

        subject_mapping = {
            old_id: new_id
            for old_id, new_id in zip(original_subjects, range(subject_offset, subject_offset + unique_subjects))
        }

        torch_ds._data[2]["subject"] = torch_ds._data[2]["subject"].map(subject_mapping)

        subject_start = subject_offset
        subject_offset += unique_subjects
        subject_end = subject_offset

        for subj_id in range(subject_start, subject_end):
            subject_to_dataset[subj_id] = dataset_name

        logger.info(f"{dataset_name}: {unique_subjects} subjects (IDs {subject_start} to {subject_end-1})")
        merged.append(torch_ds)

    total_subjects = subject_offset - 1
    hparams["n_subjects"] = total_subjects
    hparams["subject_to_dataset"] = subject_to_dataset
    logger.info(f"Total merged subjects: {total_subjects} (IDs 1-{total_subjects})")

    return ConcatDataset(merged)


def prepare_dataset_iterators(hparams):
    """
    Prepare LOSO dataset splits using manual index-based splitting.
    Pure LOSO: train on N-1 subjects, test on 1 subject.
    """
    merged_dataset = prepare_merged_dataset(hparams)

    target_subject = hparams["target_subject_idx"]

    train_indices = []
    test_indices = []

    if isinstance(merged_dataset, ConcatDataset):
        cumulative_idx = 0
        for sub_dataset in merged_dataset.datasets:
            metadata = sub_dataset._data[2]
            for local_idx in range(len(sub_dataset)):
                global_idx = cumulative_idx + local_idx
                if metadata.iloc[local_idx]["subject"] == target_subject:
                    test_indices.append(global_idx)
                else:
                    train_indices.append(global_idx)
            cumulative_idx += len(sub_dataset)
    else:
        for idx in range(len(merged_dataset)):
            sample = merged_dataset[idx]
            if sample.subject == target_subject:
                test_indices.append(idx)
            else:
                train_indices.append(idx)

    logger.info(f"Target subject: {target_subject}")
    logger.info(f"Train indices: {len(train_indices)}")
    logger.info(f"Test indices: {len(test_indices)}")

    train_dataset = Subset(merged_dataset, train_indices)
    test_dataset = Subset(merged_dataset, test_indices)

    datasets = {
        "train": train_dataset,
        "test": test_dataset
    }

    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")

    tail_path = os.path.join(
        "cross-ds-loso", f"sub-{str(target_subject).zfill(3)}"
    )
    return tail_path, datasets


def load_hparams_and_dataset_iterators(hparams_file, run_opts, overrides):
    """
    Load hyperparameters and prepare datasets.
    """
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    tail_path, datasets = prepare_dataset_iterators(hparams)
    overrides.update(n_train_examples=len(datasets["train"]))
    hparams["exp_dir"] = os.path.join(hparams["output_folder"], tail_path)

    sb.create_experiment_directory(
        experiment_directory=hparams["exp_dir"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    return hparams, datasets


if __name__ == "__main__":
    argv = sys.argv[1:]
    hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
    overrides = yaml.load(overrides, yaml.SafeLoader)
    hparams, datasets = load_hparams_and_dataset_iterators(
        hparams_file, run_opts, overrides
    )
    run_model(hparams, run_opts, datasets)
