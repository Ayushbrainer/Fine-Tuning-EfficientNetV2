"""Weight decay 1e-4: 300 epochs, plain SGD lr=0.05, per-batch cosine annealing (EfficientNetV2-S, 100x100)."""
import logging
import os
import math
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/curves", exist_ok=True)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wd1e4")

EPOCHS, LR, WD, BATCH, PAD = 300, 0.05, 1 * (10 ** -4), 128, 12
TAG = "wd1e4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("device=%s lr=%s weight_decay=%s", device, LR, WD)

# ---- data ----
trainDf = pd.read_csv("./data/train_labels.csv")
testDf = pd.read_csv("./data/test_labels.csv")
valDf = pd.read_csv("./data/val_labels.csv")
K = sorted(set(trainDf.breed) | set(testDf.breed) | set(valDf.breed))
clss_idx = {b: i for i, b in enumerate(K)}
resize = v2.Resize((100, 100), antialias=True)


def load(split, ids):
    return torch.stack([
        resize(decode_image(f"./data/{split}/images/{p}.jpg", mode=ImageReadMode.RGB))
        for p in ids
    ]).to(device)


imgs, val_imgs = load("train", trainDf.id), load("val", valDf.id)
Y_train = torch.tensor([clss_idx[b] for b in trainDf.breed], dtype=torch.long, device=device)
Y_val = torch.tensor([clss_idx[b] for b in valDf.breed], dtype=torch.long, device=device)
log.info("train=%s val=%s classes=%d", tuple(imgs.shape), tuple(val_imgs.shape), len(K))

x = imgs.float() / 255
norm = v2.Normalize(x.mean(dim=(0, 2, 3)).tolist(), x.std(dim=(0, 2, 3)).tolist())
del x


def augment(xb):
    """Per-image random horizontal flip + random crop with zero padding."""
    B, _, H, W = xb.shape
    flip = torch.rand(B, device=device) < 0.5
    xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
    xp = F.pad(xb, (PAD, PAD, PAD, PAD))
    dy = torch.randint(0, 2 * PAD + 1, (B,), device=device)
    dx = torch.randint(0, 2 * PAD + 1, (B,), device=device)
    rows = dy[:, None] + torch.arange(H, device=device)
    cols = dx[:, None] + torch.arange(W, device=device)
    b = torch.arange(B, device=device)[:, None, None]
    return xp[b, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)


def prep(xb, train=False):
    if train:
        xb = augment(xb)
    return norm(xb.float() / 255)


X_val = prep(val_imgs)

# ---- model ----
torch.manual_seed(0)
model = models.efficientnet_v2_s(weights=None).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WD)
steps_per_epoch = math.ceil(len(imgs) / BATCH)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * steps_per_epoch)  # stepped per batch


def evaluate(X, Y):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            x, y = X[i:i + BATCH], Y[i:i + BATCH]
            out = model(x)
            total_loss += criterion(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(X), correct / len(X)


# ---- train ----
hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(imgs), device=device)
    total_loss, correct = 0, 0
    for i in range(0, len(imgs), BATCH):
        idx = perm[i:i + BATCH]
        x, y = prep(imgs[idx], train=True), Y_train[idx]
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(idx)
        correct += (out.argmax(1) == y).sum().item()
    vl, va = evaluate(X_val, Y_val)
    for k, v in zip(hist, (total_loss / len(imgs), correct / len(imgs), vl, va)):
        hist[k].append(v)
    log.info("epoch %d: loss %.4f acc %.4f val_loss %.4f val_acc %.4f lr %.6f (%.0fs)",
             epoch, hist["loss"][-1], hist["acc"][-1], vl, va,
             scheduler.get_last_lr()[0], time.time() - t0)
    pd.DataFrame(hist, index=range(1, epoch + 2)).rename_axis("epoch").to_csv(f"results/metrics/metrics_{TAG}.csv")

# ---- plot ----
ep = range(1, EPOCHS + 1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(ep, hist["loss"], label="train"); a1.plot(ep, hist["val_loss"], label="val")
a1.set(xlabel="epoch", ylabel="loss", title=f"Loss vs epoch (lr={LR}, wd={WD:g})"); a1.legend()
a2.plot(ep, hist["acc"], label="train"); a2.plot(ep, hist["val_acc"], label="val")
a2.set(xlabel="epoch", ylabel="accuracy", title=f"Accuracy vs epoch (lr={LR}, wd={WD:g})"); a2.legend()
plt.tight_layout()
plt.savefig(f"results/curves/curves_{TAG}.png", dpi=120)
log.info("saved results/curves/curves_%s.png; total time %.0fs", TAG, time.time() - t0)
