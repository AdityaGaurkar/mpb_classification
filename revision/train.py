"""Train the corrected FEv4 model under the identity-level split.

Usage:  python revision/train.py
"""
import json
import os

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    RUNS_DIR,
    MPBDataset,
    fe4_cnn,
    get_device,
    load_splits,
    predict,
    seed_everything,
    train_model,
)


def main():
    seed_everything(42)
    device = get_device()
    print(f"device: {device}")

    splits = load_splits()
    train_ds = MPBDataset("train", splits)
    val_ds = MPBDataset("val", splits)
    test_ds = MPBDataset("test", splits)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = fe4_cnn().to(device)
    history, best_epoch = train_model(model, train_loader, val_loader, device)

    os.makedirs(RUNS_DIR, exist_ok=True)
    weights_path = os.path.join(RUNS_DIR, "fe4_corrected.pt")
    torch.save(model.state_dict(), weights_path)
    with open(os.path.join(RUNS_DIR, "fe4_history.json"), "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch}, f, indent=2)

    probs, labels = predict(model, test_loader, device)
    acc = float((probs.argmax(1) == labels).mean())
    print(f"\nbest epoch: {best_epoch}")
    print(f"test accuracy (identity-level split): {acc * 100:.2f}%  [n={len(labels)}]")
    print(f"weights -> {weights_path}")


if __name__ == "__main__":
    main()
