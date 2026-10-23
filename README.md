# AI6103 Homework: Learning Rate, Schedule, Weight Decay and Mixup

Training EfficientNetV2-S from scratch (random initialisation) on the Dog Breed Identification dataset (120 breeds, images resized to 100x100) with SGD (momentum 0.9), and studying how the initial learning rate, learning rate schedule, weight decay and mixup affect optimisation and generalisation.

The write-up is in `report/` (AAAI two-column format, LaTeX).

## Experiments

| Section | Setting | Epochs |
|---|---|---|
| 3 | Initial learning rate 0.5 / 0.05 / 0.001, no schedule, no weight decay | 15 |
| 4 | Best LR (0.5): constant vs cosine annealing | 300 |
| 5 | Cosine + weight decay 5e-4 and 1e-4 | 300 |
| 6 | Cosine + weight decay 5e-4 + mixup (alpha = 0.2) | 300 |

Common settings: batch size 128, seed 0, random horizontal flip (p = 0.5) and random crop (12-pixel zero padding), per-channel normalisation with training-set statistics, cross-entropy loss with label smoothing 0.1 (not required by the assignment).

### Final results (last epoch, from `train.ipynb`)

| Run | Train loss / acc | Val loss / acc |
|---|---|---|
| LR 0.5 (15 ep) | 4.5881 / 7.98% | 4.8432 / 5.41% |
| LR 0.05 (15 ep) | 4.9521 / 3.42% | 4.9880 / 3.52% |
| LR 0.001 (15 ep) | 6.8610 / 0.73% | 6.8901 / 0.52% |
| Constant LR 0.5 | 1.0381 / 99.93% | 5.5300 / 19.77% |
| Cosine | 1.0469 / 99.93% | 5.4822 / 20.74% |
| Cosine + WD 5e-4 | 1.0627 / 99.99% | 3.9221 / 34.96% |
| Cosine + WD 1e-4 | 1.0416 / 99.94% | 4.9891 / 24.98% |
| Cosine + WD 5e-4 + mixup | 1.9781 / 90.62% | 3.6858 / 25.77% |

Test accuracy of the final (mixup) model: **27.38%** (evaluated once, at the end).

Results vary a little between reruns because GPU training is not bit-for-bit deterministic, so treat small differences as noise.

## Folder layout

```
train.ipynb        Main notebook: data prep, all experiments, plots, test evaluation
data/              Dataset (train/val/test images and *_labels.csv)
src/               Script versions of the experiments (common.py, run_*.py)
jobs/              SLURM job scripts (sbatch)
logs/out, logs/err SLURM stdout / stderr files
results/metrics    Per-epoch metrics (CSV) written by the src/ scripts
results/curves     Training curves (PNG) written by the src/ scripts
models/            Saved model checkpoints
report/            LaTeX report (report.tex, refs.bib, figures/, build.sh)
```

`results/` and `models/` hold outputs of the earlier script runs. The numbers reported above come from the notebook runs.

## Running on the cluster

Training must run through SLURM on a GPU node, not on the head node. Submit jobs from the project root (log paths are relative to it):

```bash
sbatch jobs/job_jupyter.sh   # starts JupyterLab on a GPU node (see logs/out/ for the URL and port)
sbatch jobs/job_lr.sh        # Section 3 (script version)
sbatch jobs/job_sched.sh     # Section 4
sbatch jobs/job_wd.sh        # Section 5
sbatch jobs/job_mixup.sh     # Section 6
```

For the notebook, connect your kernel to the running Jupyter server and run `train.ipynb` top to bottom. The first cell should print `cuda`. Keep the kernel alive until the test-evaluation cell has run, because the test cell uses the trained model that is still in memory.

Environment: conda env `DeepEnv` (see `req.txt`).

## Building the report

```bash
bash report/build.sh   # compiles report/report.tex with tectonic from DeepEnv
```

The report is written for pdfLaTeX with the AAAI-24 style files (`aaai24.sty`, `aaai24.bst`), for example on Overleaf. The local build uses tectonic with Times-metric fonts as a preview only.
