"""Detailed holdout evaluation of the corrected FEv4 model.

Retrains on the 60/20/20 identity-level split, then reports:
accuracy (+ Wilson CI), macro-F1, per-class precision/recall/F1,
confusion matrix, within-one-stage accuracy, ordinal MAE, error distances.

Outputs: outputs/holdout_metrics.md, outputs/holdout_confusion_matrix.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    CLASSES,
    ROOT,
    RUNS_DIR,
    MPBDataset,
    confusion_matrix,
    fe4_cnn,
    get_device,
    load_splits,
    ordinal_mae,
    per_class_prf,
    predict,
    seed_everything,
    train_model,
    wilson_ci,
    within_one_accuracy,
)

OUT_DIR = os.path.join(ROOT, "outputs")


def main():
    seed_everything(42)
    device = get_device()
    splits = load_splits()
    train_loader = DataLoader(MPBDataset("train", splits), batch_size=16, shuffle=True)
    val_loader = DataLoader(MPBDataset("val", splits), batch_size=32)
    test_loader = DataLoader(MPBDataset("test", splits), batch_size=32)

    model = fe4_cnn().to(device)
    history, best_epoch = train_model(model, train_loader, val_loader, device)
    os.makedirs(RUNS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(RUNS_DIR, "fe4_corrected.pt"))

    probs, labels = predict(model, test_loader, device)
    preds = probs.argmax(1)
    n = len(labels)
    correct = int((preds == labels).sum())
    acc, lo, hi = wilson_ci(correct, n)
    f1m, f1s = None, None
    cm, prf = per_class_prf(labels, preds)
    f1m = float(np.mean([r[2] for r in prf]))
    w1 = within_one_accuracy(labels, preds)
    mae = ordinal_mae(labels, preds)
    dist = {}
    for t, p in zip(labels, preds):
        d = abs(int(t) - int(p))
        dist[d] = dist.get(d, 0) + 1

    lines = []
    lines.append("# Holdout evaluation — FEv4 (identity-level split)\n")
    lines.append(f"Test set: {n} images from {len({t[1] for t in splits['test']})} identities\n")
    lines.append(f"| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Exact accuracy | **{acc*100:.2f}%** (95% CI {lo*100:.1f}–{hi*100:.1f}%) |")
    lines.append(f"| Macro-F1 | {f1m:.3f} |")
    lines.append(f"| Within-one-stage accuracy | {w1*100:.2f}% |")
    lines.append(f"| Ordinal MAE (stages) | {mae:.3f} |")
    lines.append(f"| Best epoch (val loss) | {best_epoch} |")
    lines.append("")
    lines.append("## Per-class metrics\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for i, (p, r, f, sup) in enumerate(prf):
        lines.append(f"| {CLASSES[i]} | {p:.3f} | {r:.3f} | {f:.3f} | {sup} |")
    lines.append("")
    lines.append("## Confusion matrix (rows = true, cols = predicted)\n")
    lines.append("| true \\\\ pred | " + " | ".join(CLASSES) + " |")
    lines.append("|---|" + "---|" * len(CLASSES))
    for i, row in enumerate(cm):
        lines.append(f"| **{CLASSES[i]}** | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    lines.append("## Error distances\n")
    total_err = n - correct
    lines.append(f"- Exact: {correct}/{n}")
    for d in sorted(dist):
        if d == 0:
            continue
        lines.append(f"- |Δ| = {d} stage(s): {dist[d]} errors ({dist[d]/max(total_err,1)*100:.1f}% of errors)")
    lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "holdout_metrics.md"), "w") as fh:
        fh.write(report)
    print(report)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("FEv4 — holdout confusion matrix")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "holdout_confusion_matrix.png"), dpi=200)
    print(f"figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
