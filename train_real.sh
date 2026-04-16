#!/bin/bash
#SBATCH --account=pmc079
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=fednnunet_real
#SBATCH --output=fednnunet_real_%j.out

module purge
module load Anaconda3/2024.06
unset PYTHONHOME
unset PYTHONPATH

export nnUNet_raw=$HOME/nnunet_data/raw
export nnUNet_preprocessed=$HOME/nnunet_data/preprocessed
export nnUNet_results=$HOME/nnunet_data/results
export nnUNet_n_proc_DA=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

ENV_PY=/home/jchin/.conda/envs/fednnunet310/bin/python

cd ~/repos/FednnUNet || exit 1

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

echo "Running on $(hostname)"
echo "Using Python: $ENV_PY"
echo "PWD=$PWD"
echo "PYTHONPATH=$PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

$ENV_PY --version
$ENV_PY -c "import sys, torch, flwr, nnunetv2, fednnunet; print('exe', sys.executable); print('torch', torch.__version__); print('flwr', flwr.__version__); print('nnunetv2 ok', nnunetv2.__file__); print('fednnunet ok', getattr(fednnunet, '__file__', 'namespace-package'))"
nvidia-smi

echo "=== START FEDERATED TRAINING ==="
$ENV_PY -u -m fednnunet.run train "301 302" 3d_fullres 0 --port 8080
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "=== TRAINING FINISHED, STARTING VALIDATION ==="

# Validation-only pass. This should trigger perform_actual_validation()
# and create fold_0/validation/summary.json for the resolved trainer output.
$ENV_PY -u -m fednnunet.run train "301 302" 3d_fullres 0 --val --val_best --port 8080
VAL_EXIT=$?

if [ $VAL_EXIT -ne 0 ]; then
    echo "Validation failed with exit code $VAL_EXIT"
    exit $VAL_EXIT
fi

echo "=== VALIDATION FINISHED ==="