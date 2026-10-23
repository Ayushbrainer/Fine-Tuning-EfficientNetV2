#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=Jupyter
#SBATCH --output=logs/out/output_%x_%j.out
#SBATCH --error=logs/err/error_%x_%j.err

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate DeepEnv

cd "$SLURM_SUBMIT_DIR"

PORT=$(shuf -i 20000-30000 -n 1)
NODE=$(hostname)
echo "Node: $NODE  Port: $PORT"
echo "Tunnel from your laptop:"
echo "  ssh -N -L $PORT:$NODE:$PORT $USER@10.96.189.12"
echo "Then open http://localhost:$PORT (token is in the log lines below)"

jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
