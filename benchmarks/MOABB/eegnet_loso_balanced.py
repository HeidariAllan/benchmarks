#!/usr/bin/python
"""
EEGNet training with balanced with 3 left-out-subects.
Uses first 9 subjects from each dataset with shared channels across BNCI, Cho, and Lee datasets.
Tracks per-subject metrics.
"""

import os
import sys

# Set MNE_DATA before importing MOABB to use local datasets
os.environ['MNE_DATA'] = os.environ.get('SLURM_TMPDIR', '/tmp') + '/mne_data'

import logging
import pickle
import numpy as np
import torch
import yaml
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from functools import cached_property
import warnings
from contextlib import redirect_stdout
import pandas as pd
from sklearn import metrics

from moabb.datasets import BNCI2014_001, Cho2017, Lee2019_MI
from moabb.paradigms import MotorImagery

# MOABB local download patch
import moabb.datasets.download as moabb_dl
from urllib.parse import urlparse

def local_first_data_dl(url, sign, path=None, force_update=False, verbose=None):
    """Check if file exists locally before attempting download"""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)

    if path is None:
        path = os.environ.get('MNE_DATA', os.path.expanduser('~/mne_data'))

    # Construct local path per dataset
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

# Device setup 
DEVICE = torch.device("cuda")

# Shared channels across all three datasets
SHARED_CHANNELS = [
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'CP1', 'CP2', 'CP3', 'CP4', 'CPz',
    'Cz',
    'FC1', 'FC2', 'FC3', 'FC4',
    'Fz',
    'P1', 'P2',
    'POz', 'Pz'
] 


# ------------------------------
# Seed setting for reproducibility
# ------------------------------
def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
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
# TorchMOABBDataset wrapper for EEGNet
# ------------------------------
class TorchMOABBDataset(Dataset):
    def __init__(
        self,
        dataset,
        paradigm,
        subjects=None,
        map_labels=None,
        cache_config=None,
        pad_time=None,
        shared_channels=None,
    ):
        self.dataset = dataset
        self.paradigm = paradigm
        self.subjects = subjects or dataset.subject_list
        self.map_labels = map_labels
        self.cache_config = cache_config
        self.pad_time = pad_time
        self.shared_channels = shared_channels

    def __len__(self):
        return len(self._data[0])

    def __getitem__(self, index):
        X, y, metadata = self._data
        
        x = X[index]  # Shape: (C, T)
        if self.pad_time:
            assert self.pad_time >= x.shape[1], "Expected T <= pad_time"
            x = torch.nn.functional.pad(
                x, (0, self.pad_time - x.shape[1], 0, 0)
            )
        
        # Transpose to (T, C) and add channel dimension for EEGNet: (T, C, 1)
        x = x.transpose(0, 1).unsqueeze(-1)
        
        # Return subject_id as well for per-subject tracking
        subject_id = metadata.iloc[index]["subject"]
        
        return x, y[index], subject_id

    @cached_property
    def _data(self):
        from speechbrain.processing.signal_processing import mean_std_norm

        with warnings.catch_warnings(), open(os.devnull, "w") as _null, redirect_stdout(_null):
            warnings.simplefilter("ignore")
            X, y, metadata = self.paradigm.get_data(
                self.dataset,
                subjects=self.subjects,
                return_epochs=True,
                cache_config=self.cache_config,
            )

        # Filter to shared channels BEFORE processing
        if self.shared_channels:
            all_ch_names = X.info['ch_names']
            # Find channels that exist in both the data and our shared list
            available_shared = [ch for ch in self.shared_channels if ch in all_ch_names]
            if len(available_shared) < len(self.shared_channels):
                missing = set(self.shared_channels) - set(available_shared)
                logger.warning(f"Missing shared channels in {self.dataset.code}: {missing}")
            X = X.pick_channels(available_shared)
            logger.info(f"Filtered to {len(available_shared)} shared channels")

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

        return X, y, metadata


# ------------------------------
# Custom collate function
# ------------------------------
def collate_fn_with_subjects(batch):
    """Custom collate function that preserves subject IDs."""
    x_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    subject_batch = torch.tensor([item[2] for item in batch])
    return x_batch, y_batch, subject_batch


# ------------------------------
# Data loader preparation
# ------------------------------
def prepare_dataloaders(datasets, batch_size):
    """Create train and test data loaders."""
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_with_subjects,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn_with_subjects,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader


def compute_class_weights(train_dataset, n_classes, max_weight_ratio=2.5):
    """Compute class weights for balanced loss."""
    # Collect labels from the dataset
    if isinstance(train_dataset, Subset):
        train_labels = [
            train_dataset.dataset[idx][1].item() for idx in train_dataset.indices
        ]
    else:
        train_labels = [
            train_dataset[i][1].item() for i in range(len(train_dataset))
        ]
    
    # Compute class counts using pandas
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    
    # Ensure all classes are present
    if len(class_counts) < n_classes:
        for cls in range(n_classes):
            if cls not in class_counts:
                class_counts.loc[cls] = 0
    class_counts = class_counts.sort_index()
    
    # Compute weights
    class_weights = class_counts.max() / class_counts
    
    # Cap the maximum weight ratio
    min_weight = class_weights.min()
    max_allowed_weight = min_weight * max_weight_ratio
    class_weights = class_weights.clip(upper=max_allowed_weight)
    
    class_weights = torch.from_numpy(class_weights.values).float().to(DEVICE)
    
    logger.info(f"Class counts: {class_counts.values}")
    logger.info(f"Class weights (capped at {max_weight_ratio}x): {class_weights.cpu().numpy()}")
    return class_weights


# ------------------------------
# Training function
# ------------------------------
def run_model(hparams, run_opts, datasets):
    """Standalone training loop for balanced LOSO."""
    # Set random seed 
    seed = hparams.get("seed", 1234)
    set_seed(seed)

    logger.info(f"Using device: {DEVICE}")

    # Prepare data loaders
    train_loader, test_loader = prepare_dataloaders(
        datasets, hparams["batch_size"]
    )

    # Compute class weights
    class_weights = compute_class_weights(
        datasets["train"], 
        hparams["n_classes"],
        max_weight_ratio=hparams.get("max_weight_ratio", 2.5)
    )

    # Initialize model
    model = hparams["model"].to(DEVICE)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup optimizer and scheduler
    optimizer = hparams["optimizer"](model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams["number_of_epochs"]
    )

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
        for idx, (inputs, targets, subjects) in enumerate(train_loader):
            # Move batch to device
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            output = model(inputs)
            loss = loss_fn(output, targets, weight=class_weights)

            # Backward pass
            loss.backward()

            running_loss += loss.item()
            num_batches += 1

            # Update weights every gradient_accumulation batches
            if (idx + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

        #  optimizer step of remaining gradients
        if num_batches % gradient_accumulation != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Calculate average loss
        avg_loss = running_loss / num_batches

        # Update scheduler
        scheduler.step()
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{hparams['number_of_epochs']} - Loss: {avg_loss:.4f}")

    logger.info("Training completed, evaluating on test set...")

    # Evaluate on test set with per-subject tracking
    model.eval()
    y_true_all = []
    y_pred_all = []
    subject_ids_all = []

    # Per-subject tracking
    subject_predictions = {}
    test_subject_list = [
        hparams["target_subject_idx"],           # BNCI
        9 + hparams["target_subject_idx"],       # Cho
        18 + hparams["target_subject_idx"]       # Lee
    ]

    for subj_id in test_subject_list:
        subject_predictions[subj_id] = {'y_true': [], 'y_pred': []}

    with torch.no_grad():
        for inputs, targets, subjects in test_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            output = model(inputs)
            y_pred = torch.argmax(output, dim=-1).cpu().numpy()
            y_true = targets.cpu().numpy()
            subject_batch = subjects.numpy()

            # Aggregate all predictions
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
            subject_ids_all.extend(subject_batch)
            
            # Track per-subject predictions
            for i, subj_id in enumerate(subject_batch):
                if subj_id in subject_predictions:
                    subject_predictions[subj_id]['y_true'].append(y_true[i])
                    subject_predictions[subj_id]['y_pred'].append(y_pred[i])

    # Convert to numpy arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Compute overall metrics
    test_metrics = {}
    for metric_name, metric_func in hparams["metrics"].items():
        if metric_name == "cm":
            test_metrics[metric_name] = metric_func(y_true_all, y_pred_all).tolist()
        else:
            test_metrics[metric_name] = metric_func(y_true=y_true_all, y_pred=y_pred_all)

    logger.info(f"Overall test metrics: {test_metrics}")

    # Compute per-subject metrics
    per_subject_metrics = {}
    dataset_names = ['BNCI', 'Cho', 'Lee']

    for i, subj_id in enumerate(test_subject_list):
        y_true_subj = np.array(subject_predictions[subj_id]['y_true'])
        y_pred_subj = np.array(subject_predictions[subj_id]['y_pred'])
        dataset_name = dataset_names[i]
        
        if len(y_true_subj) > 0:  # Check if subject has data
            subj_metrics = {}
            
            # Suppress warnings for class mismatch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                for metric_name, metric_func in hparams["metrics"].items():
                    try:
                        if metric_name == "cm":
                            subj_metrics[metric_name] = metric_func(y_true_subj, y_pred_subj).tolist()
                        elif metric_name == "f1":
                            # Use macro average to handle varying number of classes gracefully
                            subj_metrics[metric_name] = metrics.f1_score(
                                y_true_subj, 
                                y_pred_subj,
                                average='macro',
                                zero_division=0
                            )
                        else:
                            subj_metrics[metric_name] = metric_func(y_true=y_true_subj, y_pred=y_pred_subj)
                    except Exception as e:
                        logger.warning(f"Could not compute {metric_name} for {dataset_name} Subject {subj_id}: {e}")
                        subj_metrics[metric_name] = np.nan
            
            per_subject_metrics[subj_id] = {
                'dataset': dataset_name,
                'metrics': subj_metrics,
                'n_samples': len(y_true_subj)
            }
            
            logger.info(f"  {dataset_name} Subject {subj_id}: "
                       f"acc={subj_metrics['acc']:.4f}, "
                       f"f1={subj_metrics['f1']:.4f}, "
                       f"n_samples={len(y_true_subj)}")
        else:
            logger.warning(f"  {dataset_name} Subject {subj_id}: No data found!")

    # Get fold info
    fold_idx = hparams["target_subject_idx"]
    test_subjects = {
        'BNCI': fold_idx,
        'Cho': 9 + fold_idx,
        'Lee': 18 + fold_idx
    }
    logger.info(f"Fold {fold_idx}/9 - Test subjects: {test_subjects}")

    # Add fold info to metrics
    test_metrics["fold_idx"] = fold_idx
    test_metrics["test_subjects"] = test_subjects
    test_metrics["per_subject_metrics"] = per_subject_metrics

    # Save results 
    save_dir = hparams["exp_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Save model 
    model_save_dir = os.path.join(save_dir, "save")
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, "model.ckpt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")

    # Save metrics 
    test_metrics_path = os.path.join(save_dir, "test_metrics.pkl")
    with open(test_metrics_path, "wb") as f:
        pickle.dump(test_metrics, f)
    logger.info(f"Saved metrics to {test_metrics_path}")

    logger.info("Training completed successfully!")
    return test_metrics


# ------------------------------
# Dataset preparation
# ------------------------------
def prepare_merged_dataset(hparams):
    """Prepare and merge BNCI, Cho, and Lee datasets with shared channels.
    Uses first 9 subjects from each dataset for balanced representation."""
    logger.info("Preparing merged dataset (BNCI+Cho+Lee) with shared channels - First 9 subjects from each...")

    pad_time = 513  
    merged = []
    subject_offset = 1
    subject_to_dataset = {}

    dataset_root = hparams["data_folder"]
    logger.info(f"Using dataset root: {dataset_root}")

    bnci = BNCI2014_001()
    cho = Cho2017()
    lee = Lee2019_MI()

    dataset_info = [
        (bnci, "BNCI2014_001", dict(left_hand=0, right_hand=1, feet=2, tongue=2), 9),
        (cho, "Cho2017", dict(left_hand=0, right_hand=1), 9),
        (lee, "Lee2019_MI", dict(left_hand=0, right_hand=1), 9)
    ]

    for ds, dataset_name, label_map, num_subjects in dataset_info:
        paradigm = MotorImagery(
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            resample=hparams["sample_rate"],
        )

        # Get only first num_subjects subjects
        available_subjects = ds.subject_list[:num_subjects]
        logger.info(f"{dataset_name}: Using first {num_subjects} subjects: {available_subjects}")

        torch_ds = TorchMOABBDataset(
            dataset=ds,
            paradigm=paradigm,
            subjects=available_subjects,
            cache_config=hparams.get("cache_config"),
            pad_time=pad_time,
            map_labels=label_map,
            shared_channels=SHARED_CHANNELS,
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
    logger.info(f"Using {len(SHARED_CHANNELS)} shared channels")

    return ConcatDataset(merged)


def prepare_dataset_iterators(hparams):
    """Prepare balanced LOSO dataset splits.
    Each fold leaves out the Nth subject from each dataset.
    
    target_subject_idx 1-9 maps to folds 1-9:
    Fold 1: Leave out subject 1 from BNCI, Cho, Lee (subjects 1, 10, 19)
    Fold 2: Leave out subject 2 from BNCI, Cho, Lee (subjects 2, 11, 20)
    ...
    Fold 9: Leave out subject 9 from BNCI, Cho, Lee (subjects 9, 18, 27)
    """
    merged_dataset = prepare_merged_dataset(hparams)

    fold_idx = hparams["target_subject_idx"]  # 1-9
    
    # Validate fold index
    if fold_idx < 1 or fold_idx > 9:
        raise ValueError(f"target_subject_idx must be between 1-9, got {fold_idx}")
    
    # Calculate which subjects to leave out
    # BNCI: subjects 1-9, Cho: subjects 10-18, Lee: subjects 19-27
    bnci_subject = fold_idx
    cho_subject = 9 + fold_idx
    lee_subject = 18 + fold_idx
    
    test_subjects = [bnci_subject, cho_subject, lee_subject]

    train_indices = []
    test_indices = []

    if isinstance(merged_dataset, ConcatDataset):
        cumulative_idx = 0
        for sub_dataset in merged_dataset.datasets:
            metadata = sub_dataset._data[2]
            for local_idx in range(len(sub_dataset)):
                global_idx = cumulative_idx + local_idx
                subject_id = metadata.iloc[local_idx]["subject"]
                if subject_id in test_subjects:
                    test_indices.append(global_idx)
                else:
                    train_indices.append(global_idx)
            cumulative_idx += len(sub_dataset)
    else:
        for idx in range(len(merged_dataset)):
            x, y = merged_dataset[idx]
            pass

    logger.info(f"Fold {fold_idx}/9 (target_subject_idx={fold_idx})")
    logger.info(f"Test subjects: BNCI={bnci_subject}, Cho={cho_subject}, Lee={lee_subject}")
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
        "leave-one-subject-out", f"sub-{str(fold_idx).zfill(2)}"
    )
    return tail_path, datasets


def load_hparams_and_dataset_iterators(hparams_file, run_opts, overrides):
    """Load hyperparameters and prepare datasets."""
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
    yaml.safe_load(overrides)
    hparams, datasets = load_hparams_and_dataset_iterators(
        hparams_file, run_opts, overrides
    )
    run_model(hparams, run_opts, datasets)
