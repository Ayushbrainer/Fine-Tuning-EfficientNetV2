"""Section 6: MixUp (alpha=0.2) on top of flip+crop augmentation. LR=0.05, 300 epochs, per-batch cosine annealing."""
import logging
import os
import math
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from common import BATCH, augment, device, load_data

TAG, LR, EPOCHS, WD, ALPHA = "mixup", 0.05, 300, 5*(10**-4), 0.2

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/curves", exist_ok=True)
log = logging.getLogger(TAG)
log.info("device=%s lr=%s epochs=%d weight_decay=%s cosine=True mixup_alpha=%s", device, LR, EPOCHS, WD, ALPHA)
imgs, Y_train, X_val, Y_val, norm, num_classes = load_data()
log.info("train=%s val=%s classes=%d", tuple(imgs.shape), tuple(X_val.shape), num_classes)

torch.manual_seed(0)
model = models.efficientnet_v2_s(weights=None, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # accepts the soft MixUp targets
optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WD)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * math.ceil(len(imgs) / BATCH))  # stepped per batch
mixup = v2.MixUp(alpha=ALPHA, num_classes=num_classes)


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
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(imgs), device=device)
    total_loss, correct = 0, 0
    for i in range(0, len(imgs), BATCH):
        idx = perm[i:i + BATCH]
        batch = {"imgs": norm(augment(imgs[idx]).float() / 255), "label": Y_train[idx]}
        batch = mixup(batch)                      # mixed images, soft labels (B, num_classes)
        x, y = batch["imgs"], batch["label"]
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(idx)
        correct += (out.argmax(1) == y.argmax(1)).sum().item()  # vs dominant class
    vl, va = evaluate()
    for k, v in zip(hist, (total_loss / len(imgs), correct / len(imgs), vl, va)):
        hist[k].append(v)
    log.info("epoch %d: loss %.4f acc %.4f val_loss %.4f val_acc %.4f lr %.6f (%.0fs)",
             epoch, hist["loss"][-1], hist["acc"][-1], vl, va, optimizer.param_groups[0]["lr"], time.time() - t0)
    pd.DataFrame(hist, index=range(1, epoch + 2)).rename_axis("epoch").to_csv(f"results/metrics/metrics_{TAG}.csv")

ep = range(1, EPOCHS + 1)
title = f"(lr={LR}, wd={WD:g}, cosine=True, mixup={ALPHA})"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(ep, hist["loss"], label="train"); a1.plot(ep, hist["val_loss"], label="val")
a1.set(xlabel="epoch", ylabel="loss", title="Loss vs epoch " + title); a1.legend()
a2.plot(ep, hist["acc"], label="train"); a2.plot(ep, hist["val_acc"], label="val")
a2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy vs epoch " + title); a2.legend()
plt.tight_layout()
plt.savefig(f"results/curves/curves_{TAG}.png", dpi=120)
log.info("saved results/curves/curves_%s.png; total time %.0fs", TAG, time.time() - t0)
