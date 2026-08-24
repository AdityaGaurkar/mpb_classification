"""5-fold identity-grouped cross-validation for FEv4 and pretrained baselines.

Every identity is tested exactly once across the 5 folds. Within each fold:
  test  = held-out identities (~20%)
  val   = ~15% of remaining identities (model selection / early stopping)
  train = the rest, offline-augmented (5 variants, train only)

Outputs:
  outputs/cv_predictions.csv   - per-image predictions for every model
  revision/runs/cv_summary.json
"""
import csv
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    MODELS,
    OG_DIR,
    ROOT,
    RUNS_DIR,
    AUG_BASE,
    MPBDataset,
    augment_identities,
    get_device,
    macro_f1,
    make_model,
    ordinal_mae,
    predict,
    seed_everything,
    train_model,
    within_one_accuracy,
)

K_FOLDS = 5
VAL_FRAC = 0.15
SEED = 42


def load_manifest():
    rows = []
    with open(os.path.join(ROOT, "outputs", "manifest.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"filename": r["filename"], "patient_id": r["patient_id"],
                         "identity_id": r["identity_id"], "label": int(r["label"])})
    return rows


def build_folds(rows):
    """Identity-level stratified fold assignment."""
    rng = random.Random(SEED)
    identities = {}
    for r in rows:
        identities.setdefault(r["identity_id"], []).append(r["label"])
    by_class = {}
    for ident, labels in identities.items():
        mode_label = max(set(labels), key=labels.count)
        by_class.setdefault(mode_label, []).append(ident)

    fold_of = {}
    for cls in sorted(by_class):
        idents = sorted(by_class[cls])
        rng.shuffle(idents)
        for i, ident in enumerate(idents):
            fold_of[ident] = i % K_FOLDS
    return fold_of


def carve_val(identities_by_class, rng):
    """Split train identities into fit/val, stratified, ~15% val."""
    fit, val = set(), set()
    for cls, idents in identities_by_class.items():
        idents = sorted(idents)
        rng.shuffle(idents)
        n_val = max(1, round(VAL_FRAC * len(idents)))
        val.update(idents[:n_val])
        fit.update(idents[n_val:])
    return fit, val


def main():
    device = get_device()
    print(f"device: {device}")
    rows = load_manifest()
    fold_of = build_folds(rows)
    identities = {}
    for r in rows:
        identities.setdefault(r["identity_id"], []).append(r)

    os.makedirs(RUNS_DIR, exist_ok=True)
    pred_path = os.path.join(ROOT, "outputs", "cv_predictions.csv")
    summary = []
    all_preds = []

    for fold in range(K_FOLDS):
        test_ids = {i for i, f in fold_of.items() if f == fold}
        train_pool = [i for i, f in fold_of.items() if f != fold]
        by_class = {}
        for i in train_pool:
            labels = [m["label"] for m in identities[i]]
            by_class.setdefault(max(set(labels), key=labels.count), []).append(i)
        rng = random.Random(SEED + fold)
        fit_ids, val_ids = carve_val(by_class, rng)

        split_map = {}
        for i in fit_ids:
            split_map[i] = "train"
        for i in val_ids:
            split_map[i] = "val"
        for i in test_ids:
            split_map[i] = "test"
        splits = {"train": [], "val": [], "test": []}
        for i, members in identities.items():
            for m in members:
                splits[split_map[i]].append((m["filename"], i, m["label"]))

        aug_dir = os.path.join(AUG_BASE, f"cv_fold{fold}_train_aug")
        n_aug = augment_identities(identities, split_map, aug_dir)

        print(f"\n===== fold {fold}: train {len(fit_ids)} ids ({n_aug} aug) | "
              f"val {len(val_ids)} | test {len(test_ids)} ids =====")

        for model_name in MODELS:
            seed_everything(SEED + fold)
            model, channels, lr = make_model(model_name)
            model = model.to(device)
            train_ds = MPBDataset("train", splits, channels=channels, aug_dir=aug_dir)
            val_ds = MPBDataset("val", splits, channels=channels)
            test_ds = MPBDataset("test", splits, channels=channels)
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32)
            test_loader = DataLoader(test_ds, batch_size=32)

            history, best_epoch = train_model(
                model, train_loader, val_loader, device, epochs=20, patience=5, max_lr=lr
            )

            probs, labels = predict(model, test_loader, device)
            preds = probs.argmax(1)
            acc = float((preds == labels).mean())
            f1_macro, _ = macro_f1(labels, preds)
            w1 = within_one_accuracy(labels, preds)
            mae = ordinal_mae(labels, preds)
            rec = {"fold": fold, "model": model_name, "n_test": len(labels),
                   "acc": acc, "macro_f1": f1_macro, "within1_acc": w1,
                   "mae": mae, "best_epoch": best_epoch}
            summary.append(rec)
            print(f"  {model_name:<20} acc {acc:.3f}  f1 {f1_macro:.3f}  "
                  f"within1 {w1:.3f}  mae {mae:.2f}  (best ep {best_epoch})")

            filenames = [s[0] for s in splits["test"]]
            for fname, ident, y, p in zip(filenames, [s[1] for s in splits["test"]], labels, preds):
                all_preds.append({"model": model_name, "fold": fold, "filename": fname,
                                  "identity_id": ident, "y_true": y, "y_pred": int(p)})

    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "fold", "filename", "identity_id",
                                          "y_true", "y_pred"])
        w.writeheader()
        w.writerows(all_preds)
    with open(os.path.join(RUNS_DIR, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== 5-fold CV summary (mean +/- std) =====")
    for model_name in MODELS:
        rs = [r for r in summary if r["model"] == model_name]
        for key in ("acc", "macro_f1", "within1_acc", "mae"):
            vals = np.array([r[key] for r in rs])
            print(f"{model_name:<20} {key:<12} {vals.mean():.3f} +/- {vals.std():.3f}")
        print()


if __name__ == "__main__":
    main()
