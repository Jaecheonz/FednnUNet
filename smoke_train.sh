#!/bin/bash
#SBATCH --account=pmc079
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=nnunet_smoke
#SBATCH --output=nnunet_smoke_%j.out

module purge
module load Anaconda3/2024.06
unset PYTHONPATH PYTHONHOME

export nnUNet_raw=$HOME/nnunet_data/raw
export nnUNet_preprocessed=$HOME/nnunet_data/preprocessed
export nnUNet_results=$HOME/nnunet_data/results

cd ~/repos/FednnUNet

echo "Running on $(hostname)"
conda run -n fednnunet310 which python
conda run -n fednnunet310 python --version
conda run -n fednnunet310 python -c "import torch, nnunetv2; print('torch', torch.__version__); print('nnunetv2 ok', nnunetv2.__file__)"
nvidia-smi
conda run -n fednnunet310 which nnUNetv2_train

CUDA_VISIBLE_DEVICES=0 conda run -n fednnunet310 nnUNetv2_train 300 2d 0