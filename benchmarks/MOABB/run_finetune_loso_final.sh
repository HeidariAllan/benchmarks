#!/bin/bash
#SBATCH --job-name=loso_final
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --account=def-ravanelm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=he_allan@live.concordia.ca

module load python/3.12.4
source ~/myenv/bin/activate

mkdir -p "$SLURM_TMPDIR/eeg_data"
mkdir -p "$SLURM_TMPDIR/.mne"
mkdir -p "$SLURM_TMPDIR/eeg_cache"
mkdir -p "$SLURM_TMPDIR/output_$SLURM_JOBID"

export HOME="$SLURM_TMPDIR"
unset _MNE_FAKE_HOME_DIR
export MNE_CONFIG_DIR="$SLURM_TMPDIR/.mne"
export MNE_DATA="$SLURM_TMPDIR/eeg_data"
export MOABB_DATASET_PATH="$SLURM_TMPDIR/eeg_data"
export MOABB_CACHE_DIR="$SLURM_TMPDIR/eeg_cache"

echo "Extracting BNCI2014_001..."
cp "$SCRATCH/eeg_datasets.tar.gz" "$SLURM_TMPDIR/eeg_data/"
tar -xzf "$SLURM_TMPDIR/eeg_data/eeg_datasets.tar.gz" \
    -C "$SLURM_TMPDIR/eeg_data" --strip-components=1

cp "$SCRATCH/eeg_results/pretrain_checkpoints/swa_model_no_bnci_44209176.ckpt" \
   "$SLURM_TMPDIR/pretrained.ckpt"

cd $SCRATCH/benchmarks/benchmarks/MOABB

python finetune_loso_final.py \
    hparams/MotorImagery/CrossDataset/SpatialEEGNetCD.yaml \
    --data_folder "$SLURM_TMPDIR/eeg_data" \
    --cached_data_folder "$SLURM_TMPDIR/eeg_cache" \
    --output_folder "$SLURM_TMPDIR/output_$SLURM_JOBID" \
    --seed 1235 \
    --pretrained_checkpoint null \
    --checkpoint "$SLURM_TMPDIR/pretrained.ckpt" \
    --projection_dim 34 \
    --cnn_temporal_kernels 40 \
    --cnn_temporal_kernelsize 61 \
    --cnn_spatial_depth_multiplier 4 \
    --cnn_septemporal_point_kernels_ratio_ 7 \
    --cnn_septemporal_kernelsize_ 12 \
    --cnn_septemporal_pool 5 \
    --spatial_focus_tau 0.05937227182082283 \
    --dropout 0.3518837062570679 \
    --channel_dropout 0.0 \
    --number_of_epochs 25 \
    --lr 0.001

mkdir -p "$SCRATCH/eeg_results/loso_final"
tar -czf "$SCRATCH/eeg_results/loso_final/results_${SLURM_JOBID}.tar.gz" \
    -C "$SLURM_TMPDIR" "output_$SLURM_JOBID"

echo "Done! Results at: $SCRATCH/eeg_results/loso_final/results_${SLURM_JOBID}.tar.gz"