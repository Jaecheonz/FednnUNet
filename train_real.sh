#!/bin/bash
#SBATCH --account=pmc079
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=fednnunet_weighted
#SBATCH --output=fednnunet_weighted_%j.out

module purge
module load Anaconda3/2024.06
unset PYTHONHOME
unset PYTHONPATH

export nnUNet_raw="$HOME/nnunet_data/raw"
export nnUNet_preprocessed="$HOME/nnunet_data/preprocessed"

# nnUNet_results is assigned below after the unique experiment
# directory has been created.
export nnUNet_n_proc_DA=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

ENV_PY=/home/jchin/.conda/envs/fednnunet310/bin/python

cd ~/repos/FednnUNet || exit 1

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

PORT=8080
SERVER_ADDRESS=127.0.0.1

# Keep this at 3 for the initial technical pilot.
# The formal G0 run must use 2000 rounds because the current
# nnU-Net trainer and PolyLR schedule are configured for 2000 epochs.
NUM_ROUNDS=${NUM_ROUNDS:-3}

EXPERIMENT_SEED=${EXPERIMENT_SEED:-2026}
STATS_EVERY=${STATS_EVERY:-10}

# Dataset 301 will provide the deterministic initial shared state.
INITIAL_DATASET_ID=${INITIAL_DATASET_ID:-301}

# G0 baseline:
# All mutually compatible parameters, including affine
# InstanceNorm parameters, are globally aggregated.
AGGREGATION_MODE=weighted

# Use a unique directory so that old checkpoints, logs and
# validation summaries cannot be reused or overwritten.
RUN_PREFIX="PILOT_G0"

if [ "$NUM_ROUNDS" -eq 2000 ]; then
    RUN_PREFIX="G0"
fi

RUN_ID="${RUN_PREFIX}_weighted_global_fold0_seed${EXPERIMENT_SEED}_${NUM_ROUNDS}r_${SLURM_JOB_ID:-local}"
RUN_DIR="$HOME/fednnunet_experiments/$RUN_ID"

mkdir -p "$RUN_DIR"

# Give this experiment its own isolated nnU-Net results directory.
export nnUNet_results="$RUN_DIR/nnUNet_results"
mkdir -p "$nnUNet_results"

# run_training.py also reads the seed from the environment.
export EXPERIMENT_SEED

echo "Running on $(hostname)"
echo "Using Python: $ENV_PY"
echo "PWD=$PWD"
echo "PYTHONPATH=$PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
echo "PORT=$PORT"
echo "SERVER_ADDRESS=$SERVER_ADDRESS"
echo "NUM_ROUNDS=$NUM_ROUNDS"
echo "EXPERIMENT_SEED=$EXPERIMENT_SEED"
echo "STATS_EVERY=$STATS_EVERY"
echo "INITIAL_DATASET_ID=$INITIAL_DATASET_ID"
echo "AGGREGATION_MODE=$AGGREGATION_MODE"
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"

$ENV_PY --version
$ENV_PY -c "import sys, torch, flwr, nnunetv2, fednnunet; print('exe', sys.executable); print('torch', torch.__version__); print('flwr', flwr.__version__); print('nnunetv2 ok', nnunetv2.__file__); print('fednnunet ok', getattr(fednnunet, '__file__', 'namespace-package'))"
nvidia-smi

echo "=== PYTHON COMPILE CHECKS ==="
$ENV_PY -m py_compile \
    fednnunet/server.py \
    fednnunet/client.py \
    fednnunet/run.py \
    fednnunet/client_entrypoints.py \
    fednnunet/run_training.py \
    fednnunet/experiment_utils.py

COMPILE_EXIT=$?

if [ "$COMPILE_EXIT" -ne 0 ]; then
    echo "Python compile checks failed with exit code $COMPILE_EXIT"
    exit "$COMPILE_EXIT"
fi

echo "=== START FEDERATED TRAINING ==="

$ENV_PY -u -m fednnunet.run train "301 302" 3d_fullres 0 \
    --port "$PORT" \
    --server_address "$SERVER_ADDRESS" \
    --num_rounds "$NUM_ROUNDS" \
    --aggregation_mode "$AGGREGATION_MODE" \
    --seed "$EXPERIMENT_SEED" \
    --run_dir "$RUN_DIR" \
    --stats_every "$STATS_EVERY" \
    --initial_dataset_id "$INITIAL_DATASET_ID"

TRAIN_EXIT=$?

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit "$TRAIN_EXIT"
fi

echo "=== FEDERATED TRAINING AND FINAL VALIDATION FINISHED ==="
echo "Experiment directory: $RUN_DIR"

EXPECTED_SERVER_CHECKPOINT="$RUN_DIR/fold_0/server_shared_final.pth"
EXPECTED_CLIENT301_SUMMARY="$RUN_DIR/fold_0/client_301/final_post_aggregation/summary.json"
EXPECTED_CLIENT302_SUMMARY="$RUN_DIR/fold_0/client_302/final_post_aggregation/summary.json"

MISSING_OUTPUT=0

if [ ! -f "$EXPECTED_SERVER_CHECKPOINT" ]; then
    echo "Missing expected server checkpoint:"
    echo "  $EXPECTED_SERVER_CHECKPOINT"
    MISSING_OUTPUT=1
fi

if [ ! -f "$EXPECTED_CLIENT301_SUMMARY" ]; then
    echo "Missing expected Dataset 301 summary:"
    echo "  $EXPECTED_CLIENT301_SUMMARY"
    MISSING_OUTPUT=1
fi

if [ ! -f "$EXPECTED_CLIENT302_SUMMARY" ]; then
    echo "Missing expected Dataset 302 summary:"
    echo "  $EXPECTED_CLIENT302_SUMMARY"
    MISSING_OUTPUT=1
fi

if [ "$MISSING_OUTPUT" -ne 0 ]; then
    echo "The federated command exited successfully, but required outputs are missing."
    exit 1
fi

echo "Required G0 outputs were created successfully."