"""Shared pipeline for the run_*.py experiments (EfficientNetV2-S, 100x100, GPU-resident data)."""
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
BATCH, PAD = 128, 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    """Returns imgs (uint8), Y_train, X_val (normalised), Y_val, norm, num_classes."""
    dfs = {s: pd.read_csv(f"./data/{s}_labels.csv") for s in ("train", "test", "val")}
    K = sorted(set().union(*(set(d.breed) for d in dfs.values())))
    idx = {b: i for i, b in enumerate(K)}
    resize = v2.Resize((100, 100), antialias=True)

    def load(split):
        return torch.stack([
            resize(decode_image(f"./data/{split}/images/{p}.jpg", mode=ImageReadMode.RGB))
            for p in dfs[split].id
        ]).to(device)

    def labels(split):
        return torch.tensor([idx[b] for b in dfs[split].breed], dtype=torch.long, device=device)

    imgs, val_imgs = load("train"), load("val")
    x = imgs.float() / 255
    norm = v2.Normalize(x.mean(dim=(0, 2, 3)).tolist(), x.std(dim=(0, 2, 3)).tolist())
    del x
    return imgs, labels("train"), norm(val_imgs.float() / 255), labels("val"), norm, len(K)


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


def run(tag, lr, epochs, weight_decay=0.0, cosine=False):
    """Train a fresh model (seed 0, SGD, label smoothing 0.1), log per epoch, save metrics csv + curves png."""
    log = logging.getLogger(tag)
    log.info("device=%s lr=%s epochs=%d weight_decay=%s cosine=%s", device, lr, epochs, weight_decay, cosine)
    imgs, Y_train, X_val, Y_val, norm, _ = load_data()
    log.info("train=%s val=%s", tuple(imgs.shape), tuple(X_val.shape))

    torch.manual_seed(0)
    model = models.efficientnet_v2_s(weights=None).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (CosineAnnealingLR(optimizer, T_max=epochs * math.ceil(len(imgs) / BATCH))  # stepped per batch
                 if cosine else None)

    def evaluate():
        model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for i in range(0, len(X_val), BATCH):
                x, y = X_val[i:i + BATCH], Y_val[i:i + BATCH]
                out = model(x)
                total_loss += criterion(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
        return total_loss / len(X_val), correct / len(X_val)

    hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(imgs), device=device)
        total_loss, correct = 0, 0
        for i in range(0, len(imgs), BATCH):
            idx = perm[i:i + BATCH]
            x, y = norm(augment(imgs[idx]).float() / 255), Y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == y).sum().item()
        vl, va = evaluate()
        for k, v in zip(hist, (total_loss / len(imgs), correct / len(imgs), vl, va)):
            hist[k].append(v)
        log.info("epoch %d: loss %.4f acc %.4f val_loss %.4f val_acc %.4f lr %.6f (%.0fs)",
                 epoch, hist["loss"][-1], hist["acc"][-1], vl, va, optimizer.param_groups[0]["lr"], time.time() - t0)
        pd.DataFrame(hist, index=range(1, epoch + 2)).rename_axis("epoch").to_csv(f"results/metrics/metrics_{tag}.csv")

    ep = range(1, epochs + 1)
    title = f"(lr={lr}, wd={weight_decay:g}, cosine={cosine})"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, hist["loss"], label="train"); a1.plot(ep, hist["val_loss"], label="val")
    a1.set(xlabel="epoch", ylabel="loss", title="Loss vs epoch " + title); a1.legend()
    a2.plot(ep, hist["acc"], label="train"); a2.plot(ep, hist["val_acc"], label="val")
    a2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy vs epoch " + title); a2.legend()
    plt.tight_layout()
    plt.savefig(f"results/curves/curves_{tag}.png", dpi=120)
    log.info("saved results/curves/curves_%s.png; total time %.0fs", tag, time.time() - t0)
