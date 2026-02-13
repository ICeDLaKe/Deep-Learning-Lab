import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
import random

def visualize_10_per_class(dataset, save_dir, filename="10_per_class.png"):
    os.makedirs(save_dir, exist_ok=True)
    per_class = defaultdict(list)
    for i in range(len(dataset)):
        _, lbl = dataset[i]
        per_class[lbl].append(i)
    sampled = {}
    for cls in range(10):
        idxs = per_class[cls]
        random.shuffle(idxs)
        sampled[cls] = idxs[:10]
    fig, axes = plt.subplots(10,10, figsize=(12,12))
    for cls in range(10):
        for j in range(10):
            ax = axes[cls,j]
            img, _ = dataset[sampled[cls][j]]
            img = img.permute(1,2,0).numpy()
            ax.imshow(np.clip(img,0,1))
            ax.set_xticks([]); ax.set_yticks([])
            if j==0:
                ax.set_ylabel(f"Class {cls}", rotation=0, labelpad=20, va='center')
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def plot_history(history_csv, save_prefix):
    import csv
    epochs, tr_loss, tr_acc, va_loss, va_acc = [],[],[],[],[]
    with open(history_csv,'r') as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            tr_acc.append(float(row["train_acc"]))
            va_loss.append(float(row["val_loss"]))
            va_acc.append(float(row["val_acc"]))
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    plt.figure(figsize=(7,5)); plt.plot(epochs, tr_loss, label="Train Loss"); plt.plot(epochs, va_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.savefig(f"{save_prefix}_loss.png", dpi=200); plt.close()
    plt.figure(figsize=(7,5)); plt.plot(epochs, tr_acc, label="Train Acc"); plt.plot(epochs, va_acc, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend(); plt.grid(True); plt.savefig(f"{save_prefix}_acc.png", dpi=200); plt.close()
