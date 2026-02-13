import torch
import os
import csv
from tqdm import tqdm
from copy import deepcopy

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs
    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs
    return total_loss / n, total_acc / n

def run_training(model, train_loader, val_loader, cfg, out_dir):
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    os.makedirs(out_dir, exist_ok=True)
    history = []
    best_val = -1.0
    best_state = None
    patience = cfg.get("early_stop_patience", 5)
    wait = 0

    for epoch in range(1, cfg.get("epochs", 30) + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device, criterion)
        va_loss, va_acc = eval_epoch(model, val_loader, device, criterion)
        history.append([epoch, tr_loss, tr_acc, va_loss, va_acc])
        print(f"Epoch {epoch:03d}: Train Acc={tr_acc:.4f}, Val Acc={va_acc:.4f}")

        # checkpoint
        if va_acc > best_val:
            best_val = va_acc
            best_state = deepcopy(model.state_dict())
            wait = 0
            torch.save(best_state, os.path.join(out_dir, "best_model.pt"))
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping after {epoch} epochs (patience={patience}).")
            break

    # save history csv
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(history)

    return history, best_val
