#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=TrainNB
#SBATCH --output=logs/out/output_%x_%j.out
#SBATCH --error=logs/err/error_%x_%j.err

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate DeepEnv

cd "$SLURM_SUBMIT_DIR"

python -c "import torch; print('cuda available:', torch.cuda.is_available())"

jupyter nbconvert --to notebook --execute train.ipynb \
    --output train_executed.ipynb \
    --ExecutePreprocessor.timeout=-1
