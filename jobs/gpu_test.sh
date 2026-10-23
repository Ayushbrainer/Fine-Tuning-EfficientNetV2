#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --job-name=GpuTest
#SBATCH --output=logs/out/output_%x_%j.out
#SBATCH --error=logs/err/error_%x_%j.err

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate DeepEnv

hostname
python - <<'EOF'
import torch, torchvision
from torchvision.io import decode_image, ImageReadMode
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("matmul ok on", (x @ x).device)
EOF
