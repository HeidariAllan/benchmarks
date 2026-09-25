#!/usr/bin/python
"""
Full 9-fold LOSO on BNCI2014_001.

Conditions:
  - Adapter:  pretrained backbone (frozen) + adapter (bottleneck=6, lr=0.0001, 60ep)
  - Scratch:  random init, lr=0.0001, 40 epochs

Both use the standard recipe for adapter:
  1. Load pretrained backbone (frozen)
  2. Reinitialize fc_out randomly
  3. Attach adapter after conv_module (bottleneck=6, zero-init)
  4. Train adapter + fc_out only
"""

import os
import sys
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import logging
import warnings
from contextlib import redirect_stdout
from urllib.parse import urlparse
import pandas as pd
from functools import cached_property

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, Subset
from sklearn import metrics as sklearn_metrics

# ============================================================================
# MOABB PATCH
# ============================================================================
import moabb.datasets.download as moabb_dl

def local_first_data_dl(url, sign, path=None, force_update=False, verbose=None):
    parsed = urlparse(url)
    if path is None:
        path = os.environ.get('MNE_DATA', os.path.expanduser('~/mne_data'))
    local_path = os.path.join(path, 'MNE-bnci-data', parsed.path.lstrip('/'))
    if os.path.exists(local_path) and not force_update:
        return local_path
    raise FileNotFoundError(f"File not found locally: {local_path}")

moabb_dl.data_dl = local_first_data_dl

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SUBJECTS = 9

# Best configs from sweep
ADAPTER_BOTTLENECK = 6
ADAPTER_LR         = 0.0001
ADAPTER_EPOCHS     = 60

SCRATCH_LR     = 0.0001
SCRATCH_EPOCHS = 40


# ============================================================================
# Adapter module
# ============================================================================
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_dim, d_model)
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)  # zero-init → identity at start
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class ConvModuleAdapterHook:
    def __init__(self, adapter):
        self.adapter = adapter

    def __call__(self, module, input, output):
        if not isinstance(output, torch.Tensor):
            return output
        if output.dim() == 2:
            return self.adapter(output)
        elif output.dim() == 3:
            B, d, T = output.shape
            x = output.permute(0, 2, 1).reshape(B * T, d)
            x = self.adapter(x)
            return x.reshape(B, T, d).permute(0, 2, 1)
        elif output.dim() == 4:
            B, d, T, W = output.shape
            x = output.reshape(B, d, T * W).permute(0, 2, 1).reshape(B * T * W, d)
            x = self.adapter(x)
            return x.reshape(B, T * W, d).permute(0, 2, 1).reshape(B, d, T, W)
        return output


# ============================================================================
# Dataset
# ============================================================================
class TorchMOABBDataset(Dataset):
    def __init__(self, dataset, paradigm, subjects=None, map_labels=None,
                 cache_config=None, head_sphere=((0, 0, 0.04), 0.09), pad_time=None):
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
            if x.shape[1] > self.pad_time:
                x = x[:, :self.pad_time]
            elif x.shape[1] < self.pad_time:
                x = torch.nn.functional.pad(x, (0, self.pad_time - x.shape[1], 0, 0))
        return Data(x=x, y=y[index], pos=ch_positions, subject=meta["subject"],
                   session=meta["session"], run=meta["run"])

    @cached_property
    def _data(self):
        from speechbrain.processing.signal_processing import mean_std_norm
        with warnings.catch_warnings(), redirect_stdout(None):
            warnings.simplefilter("ignore")
            X, y, metadata = self.paradigm.get_data(
                self.dataset, subjects=self.subjects,
                return_epochs=True, cache_config=self.cache_config
            )
        offset = torch.tensor(self.head_sphere[0])
        radius = self.head_sphere[1]
        ch_positions = (
            torch.from_numpy(
                np.array(list(X.info.get_montage().get_positions()["ch_pos"].values()))
            ).sub_(offset).div_(radius).float().contiguous()
        )
        ch_positions = ch_positions * 5.0
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


def get_loso_indices(bnci_ds, held_out_subject):
    metadata = bnci_ds._data[2]
    train_indices, test_indices = [], []
    for i in range(len(bnci_ds)):
        subj = metadata.iloc[i]["subject"]
        if subj == held_out_subject:
            test_indices.append(i)
        else:
            train_indices.append(i)
    return train_indices, test_indices


def get_class_weights(bnci_ds, train_indices, n_classes, device):
    train_labels = [bnci_ds[i].y.item() for i in train_indices]
    label_counts = pd.Series(train_labels).value_counts().sort_index()
    for cls in range(n_classes):
        if cls not in label_counts:
            label_counts[cls] = 0
    label_counts = label_counts.sort_index()
    class_weights = (
        label_counts.max() / label_counts.replace(0, label_counts.max())
    ).clip(upper=2.5)
    return torch.tensor(class_weights.values, dtype=torch.float).to(device)


def train_loop(model, loader, optimizer, loss_fn, class_weights, n_epochs,
               extra_modules=None):
    for epoch in range(1, n_epochs + 1):
        model.train()
        if extra_modules:
            for m in extra_modules:
                m.train()
        running_loss = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = loss_fn(output, batch.y, weight=class_weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += 1
        if epoch % 10 == 0 or epoch == n_epochs:
            logger.info(f"      Epoch {epoch:02d}/{n_epochs} | loss={running_loss/max(n,1):.4f}")


def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            output = model(batch)
            y_pred.extend(torch.argmax(output, dim=-1).cpu().numpy())
            y_true.extend(batch.y.cpu().numpy())
    acc = sklearn_metrics.accuracy_score(y_true, y_pred)
    f1  = sklearn_metrics.f1_score(y_true, y_pred, average='macro')
    return acc, f1


# ============================================================================
# One fold — adapter
# ============================================================================
def run_adapter_fold(model_template, hparams, bnci_ds, pretrained_state,
                     held_out_subject, conv_d):
    train_indices, test_indices = get_loso_indices(bnci_ds, held_out_subject)
    train_loader = DataLoader(
        Subset(bnci_ds, train_indices),
        batch_size=hparams["batch_size"], shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        Subset(bnci_ds, test_indices),
        batch_size=hparams["batch_size"], shuffle=False, num_workers=0
    )
    class_weights = get_class_weights(
        bnci_ds, train_indices, hparams["n_classes"], DEVICE
    )
    loss_fn = hparams["loss"]

    # Load pretrained backbone
    model = copy.deepcopy(model_template).to(DEVICE)
    model.load_state_dict(pretrained_state, strict=False)

    # Reinitialize fc_out
    torch.nn.init.xavier_uniform_(model.dense_module.fc_out.w.weight)
    torch.nn.init.zeros_(model.dense_module.fc_out.w.bias)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    for param in model.dense_module.fc_out.parameters():
        param.requires_grad = True

    # Attach adapter
    adapter = Adapter(d_model=conv_d, bottleneck_dim=ADAPTER_BOTTLENECK).to(DEVICE)
    hook_fn = ConvModuleAdapterHook(adapter)
    handle  = model.conv_module.register_forward_hook(hook_fn)

    optimizer = torch.optim.Adam(
        list(adapter.parameters()) +
        list(model.dense_module.fc_out.parameters()),
        lr=ADAPTER_LR
    )

    train_loop(model, train_loader, optimizer, loss_fn, class_weights,
               ADAPTER_EPOCHS, extra_modules=[adapter])

    model.eval()
    adapter.eval()
    acc, f1 = evaluate(model, test_loader)
    handle.remove()

    return acc, f1


# ============================================================================
# One fold — scratch
# ============================================================================
def run_scratch_fold(model_template, hparams, bnci_ds, held_out_subject):
    train_indices, test_indices = get_loso_indices(bnci_ds, held_out_subject)
    train_loader = DataLoader(
        Subset(bnci_ds, train_indices),
        batch_size=hparams["batch_size"], shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        Subset(bnci_ds, test_indices),
        batch_size=hparams["batch_size"], shuffle=False, num_workers=0
    )
    class_weights = get_class_weights(
        bnci_ds, train_indices, hparams["n_classes"], DEVICE
    )
    loss_fn = hparams["loss"]

    model = copy.deepcopy(model_template).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=SCRATCH_LR)

    train_loop(model, train_loader, optimizer, loss_fn, class_weights,
               SCRATCH_EPOCHS)

    acc, f1 = evaluate(model, test_loader)
    return acc, f1


# ============================================================================
# Probe conv_module output dim
# ============================================================================
def probe_conv_d(model, bnci_ds, train_indices, batch_size):
    shapes = []
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            shapes.append(output.shape)
    handle = model.conv_module.register_forward_hook(hook)
    loader = DataLoader(
        Subset(bnci_ds, train_indices[:batch_size]),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            try:
                model(batch)
            except Exception:
                pass
            break
    handle.remove()
    return shapes[0][1] if shapes else 281


# ============================================================================
# Full LOSO
# ============================================================================
def run_loso(hparams, pretrained_checkpoint_path):
    logger.info("Loading BNCI2014_001 (all 9 subjects)...")
    paradigm = MotorImagery(
        fmin=hparams["fmin"], fmax=hparams["fmax"], resample=hparams["sample_rate"]
    )
    bnci_ds = TorchMOABBDataset(
        dataset=BNCI2014_001(), paradigm=paradigm,
        subjects=list(range(1, N_SUBJECTS + 1)),
        cache_config=hparams.get("cache_config"),
        pad_time=hparams["T"],
        map_labels={'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 2}
    )
    logger.info(f"Loaded {len(bnci_ds)} BNCI trials")

    logger.info(f"Loading checkpoint: {pretrained_checkpoint_path}")
    pretrained_state = torch.load(pretrained_checkpoint_path, map_location=DEVICE)
    if any(k.startswith("module.") for k in pretrained_state.keys()):
        logger.info("Unwrapping SWA AveragedModel...")
        pretrained_state = {
            k.replace("module.", "", 1): v
            for k, v in pretrained_state.items()
            if k.startswith("module.")
        }

    model_template = hparams["model"]

    # Probe conv_module output dim
    probe_model = copy.deepcopy(model_template).to(DEVICE)
    probe_model.load_state_dict(pretrained_state, strict=False)
    train_idx_s1, _ = get_loso_indices(bnci_ds, 1)
    conv_d = probe_conv_d(probe_model, bnci_ds, train_idx_s1, hparams["batch_size"])
    del probe_model
    logger.info(f"conv_module output dim: {conv_d}")

    results = {
        'adapter': {'per_subject': {}, 'y_true': [], 'y_pred': []},
        'scratch': {'per_subject': {}, 'y_true': [], 'y_pred': []},
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"FULL LOSO — BNCI2014_001 ({N_SUBJECTS} subjects)")
    logger.info(f"Adapter:  bottleneck={ADAPTER_BOTTLENECK}, lr={ADAPTER_LR}, epochs={ADAPTER_EPOCHS}")
    logger.info(f"Scratch:  lr={SCRATCH_LR}, epochs={SCRATCH_EPOCHS}")
    logger.info(f"{'='*80}")

    for held_out in range(1, N_SUBJECTS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {held_out}/{N_SUBJECTS} — held-out subject {held_out}")
        logger.info(f"{'='*60}")

        # --- Adapter ---
        logger.info(f"  [adapter] bottleneck={ADAPTER_BOTTLENECK}, lr={ADAPTER_LR}, epochs={ADAPTER_EPOCHS}")
        acc_a, f1_a = run_adapter_fold(
            model_template, hparams, bnci_ds, pretrained_state, held_out, conv_d
        )
        results['adapter']['per_subject'][held_out] = {'acc': acc_a, 'f1': f1_a}
        results['adapter']['y_true'].extend(
            [bnci_ds[i].y.item() for i in get_loso_indices(bnci_ds, held_out)[1]]
        )
        logger.info(f"  [adapter] Subject {held_out} → acc={acc_a:.4f} | f1={f1_a:.4f}")

        # --- Scratch ---
        logger.info(f"  [scratch] lr={SCRATCH_LR}, epochs={SCRATCH_EPOCHS}")
        acc_s, f1_s = run_scratch_fold(
            model_template, hparams, bnci_ds, held_out
        )
        results['scratch']['per_subject'][held_out] = {'acc': acc_s, 'f1': f1_s}
        logger.info(f"  [scratch] Subject {held_out} → acc={acc_s:.4f} | f1={f1_s:.4f}")

        logger.info(f"  Δacc = {acc_a - acc_s:+.4f} (adapter - scratch)")

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*80}")
    logger.info("FINAL RESULTS — BNCI2014_001 LOSO")
    logger.info(f"{'='*80}")

    for condition in ['adapter', 'scratch']:
        per_subj = results[condition]['per_subject']
        accs = [v['acc'] for v in per_subj.values()]
        f1s  = [v['f1']  for v in per_subj.values()]

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        mean_f1  = np.mean(f1s)
        std_f1   = np.std(f1s)

        results[condition]['overall'] = {
            'mean_acc': mean_acc, 'std_acc': std_acc,
            'mean_f1':  mean_f1,  'std_f1':  std_f1,
        }

        label = "Adapter (pretrained)" if condition == 'adapter' else "Scratch (random init)"
        logger.info(f"\n[{label}]")
        logger.info(f"  acc: {mean_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"  f1:  {mean_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"  Per-subject:")
        for subj, v in per_subj.items():
            logger.info(f"    Subject {subj:2d}: acc={v['acc']:.4f} | f1={v['f1']:.4f}")

    delta_acc = results['adapter']['overall']['mean_acc'] - results['scratch']['overall']['mean_acc']
    delta_f1  = results['adapter']['overall']['mean_f1']  - results['scratch']['overall']['mean_f1']
    logger.info(f"\n{'='*80}")
    logger.info(f"Transfer gain: Δacc={delta_acc:+.4f} | Δf1={delta_f1:+.4f}")
    logger.info(f"{'='*80}")

    # Save
    save_dir = os.path.join(hparams["output_folder"], "loso_save")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "loso_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {save_dir}/loso_results.pkl")

    return results


if __name__ == "__main__":
    argv = sys.argv[1:]

    pretrained_checkpoint = None
    filtered_argv = []
    i = 0
    while i < len(argv):
        if argv[i] == "--checkpoint":
            pretrained_checkpoint = argv[i + 1]
            i += 2
        else:
            filtered_argv.append(argv[i])
            i += 1

    if pretrained_checkpoint is None:
        raise ValueError("Must provide --checkpoint /path/to/swa_model.ckpt")

    hparams_file, run_opts, overrides = sb.core.parse_arguments(filtered_argv)
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    import random
    seed = hparams.get("seed", 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_loso(hparams, pretrained_checkpoint)