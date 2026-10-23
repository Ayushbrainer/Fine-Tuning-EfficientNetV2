#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --job-name=WeightDecay
#SBATCH --output=logs/out/output_%x_%j.out
#SBATCH --error=logs/err/error_%x_%j.err

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate DeepEnv

cd "$SLURM_SUBMIT_DIR"

python src/run_wd5e4.py
python src/run_wd1e4.py
