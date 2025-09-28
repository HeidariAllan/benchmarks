#!/usr/bin/python
"""
Cross-dataset training with leave-one-subject-out (LOSO).
Merges BNCI2014_001, Cho2017, and Lee2019_MI into one unified dataset with
unique subject IDs and runs the standard SpeechBrain training pipeline.

Usage:
    python cross_ds_train.py hparams/MotorImagery/EEGNet.yaml \
        --data_folder=eeg_data \
        --cached_data_folder=eeg_pickled_data \
        --output_folder=results/MotorImagery/EEGNet \
        --data_iterator_name=leave-one-subject-out \
        --target_subject_idx=0
"""

import logging
import os
import pickle
import sys

import numpy as np
import speechbrain as sb
import torch
import yaml
from hyperpyyaml import load_hyperpyyaml
from torch.nn import init
from torch_geometric.data import Batch, Data
from torch.utils.data import ConcatDataset, Dataset
from functools import cached_property
import warnings
from contextlib import redirect_stdout
import pandas as pd

from moabb.datasets import BNCI2014_001, Cho2017, Lee2019_MI
from moabb.paradigms import MotorImagery
from utils.graph_iterators import LeaveOneSubjectOut


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
# SpeechBrain Brain class
# ------------------------------
class MOABBBrain(sb.Brain):
    def init_model(self, model):
        for mod in model.modules():
            if hasattr(mod, "weight"):
                if "Norm" not in mod.__class__.__name__:
                    init.xavier_uniform_(mod.weight, gain=1)
                else:
                    init.constant_(mod.weight, 1)
            if hasattr(mod, "bias") and mod.bias is not None:
                init.constant_(mod.bias, 0)

    def compute_forward(self, batch, stage):
        inputs = batch.to(self.device)

        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "augment")
            and self.hparams.repeat_augment > 0
        ):
            aug, _ = self.hparams.augment(
                inputs.x.unsqueeze(-1),
                lengths=torch.ones(inputs.x.shape[0], device=self.device),
            )
            if self.hparams.augment.concat_original:
                inputs = inputs.concat(inputs)
            inputs.x = aug.squeeze(-1)

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "graph_augment"):
            inputs = self.hparams.graph_augment(inputs)

        if hasattr(self.hparams, "normalize"):
            inputs.x = self.hparams.normalize(inputs.x)
        return self.modules.model(inputs)

    def compute_objectives(self, predictions, batch, stage):
        targets = batch.y.to(self.device)
        N_augments = int(predictions.shape[0] / targets.shape[0])
        targets = torch.cat(N_augments * [targets], dim=0)

        loss = self.hparams.loss(
            predictions,
            targets,
            weight=torch.FloatTensor(self.hparams.class_weights).to(self.device),
        )
        if stage != sb.Stage.TRAIN:
            tmp_preds = torch.exp(predictions)
            self.preds.extend(tmp_preds.detach().cpu().numpy())
            self.targets.extend(batch.y.cpu().numpy())
        else:
            if hasattr(self.hparams, "lr_annealing"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)
        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.preds, self.targets = [], []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            preds = np.array(self.preds)
            y_pred = np.argmax(preds, axis=-1)
            y_true = self.targets
            self.last_eval_stats = {"loss": stage_loss}
            for metric_key in self.hparams.metrics.keys():
                self.last_eval_stats[metric_key] = self.hparams.metrics[metric_key](
                    y_true=y_true, y_pred=y_pred
                )


# ------------------------------
# Training pipeline
# ------------------------------
def run_experiment(hparams, run_opts, datasets):
    ys = Batch.from_data_list(datasets["train"].dataset).y
    idx_examples = np.arange(ys.shape[0])
    n_examples_perclass = [
        idx_examples[np.where(ys == c)[0]].shape[0]
        for c in range(hparams["n_classes"])
    ]
    class_weights = np.array(n_examples_perclass).max() / np.array(
        n_examples_perclass
    )
    hparams["class_weights"] = class_weights

    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=os.path.join(hparams["exp_dir"], "save"),
        recoverables={"model": hparams["model"], "counter": hparams["epoch_counter"]},
    )
    hparams["train_logger"] = sb.utils.train_logger.FileTrainLogger(
        save_file=os.path.join(hparams["exp_dir"], "train_log.txt")
    )

    brain = MOABBBrain(
        modules={"model": hparams["model"]},
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        progressbar=False,
    )
    brain.evaluate(datasets["test"], progressbar=False)


def prepare_dataset_iterators(hparams):
    print("Preparing merged dataset (BNCI+Cho+Lee)...")

    paradigm = MotorImagery(
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
        resample=hparams["sample_rate"],
    )
    pad_time = 640

    merged = []
    subject_offset = 0

    # Dataset root provided by sbatch (from $SLURM_TMPDIR/eeg_data)
    dataset_root = hparams["data_folder"]

    # BNCI2014-001
    bnci = BNCI2014_001()
    bnci.dataset_path = os.path.join(dataset_root, "BNCI2014-001")

    # Cho2017
    cho = Cho2017()
    cho.dataset_path = os.path.join(dataset_root, "Cho2017")

    # Lee2019_MI
    lee = Lee2019_MI()
    lee.dataset_path = os.path.join(dataset_root, "Lee2019_MI")

    for ds in [bnci, cho, lee]:
        torch_ds = TorchMOABBDataset(
            dataset=ds,
            paradigm=paradigm,
            cache_config=hparams.get("cache_config"),
            pad_time=pad_time,
        )
        torch_ds._data[2]["subject"] += subject_offset
        subject_offset = torch_ds._data[2]["subject"].max() + 1
        merged.append(torch_ds)

    total_subjects = subject_offset
    hparams["n_subjects"] = total_subjects
    print(f"Total merged subjects: {total_subjects}")

    data_iterator = LeaveOneSubjectOut(
        datasets=[ConcatDataset(merged)],
        resample=hparams["sample_rate"],
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
        tmin=hparams["tmin"],
        tmax=hparams["tmax"],
        events=hparams["events_to_load"],
        valid_ratio=hparams["valid_ratio"],
        target_subjects=hparams["target_subject_idx"] + 1,
        target_sessions=hparams["target_session_idx"],
    )

    datasets = data_iterator.prepare(
        data_folder=hparams["data_folder"],
        cached_data_folder=hparams["cached_data_folder"],
        batch_size=hparams["batch_size"],
    )
    tail_path = os.path.join(
        "cross-ds-loso", f"sub-{str(hparams['target_subject_idx']+1).zfill(3)}"
    )
    return tail_path, datasets


def load_hparams_and_dataset_iterators(hparams_file, run_opts, overrides):
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    tail_path, datasets = prepare_dataset_iterators(hparams)
    overrides.update(n_train_examples=len(datasets["train"].dataset))
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
    run_experiment(hparams, run_opts, datasets)
