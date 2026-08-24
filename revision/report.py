"""Aggregate 5-fold CV predictions into tables, figures and a results summary.

Reads:  outputs/cv_predictions.csv, revision/runs/cv_summary.json
Writes: outputs/cv_metrics.csv, outputs/pooled_confusion_<model>.png,
        outputs/cv_summary_table.md, outputs/RESULTS_SUMMARY.md
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    CLASSES,
    MODELS,
    ROOT,
    confusion_matrix,
    ordinal_mae,
    per_class_prf,
    wilson_ci,
    within_one_accuracy,
)

OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_LABELS = {
    "fe4": "Custom CNN (FEv4, from scratch)",
    "squeezenet1_1": "SqueezeNet 1.1 (ImageNet)",
    "mobilenet_v3_small": "MobileNetV3-Small (ImageNet)",
    "efficientnet_b0": "EfficientNet-B0 (ImageNet)",
}


def load_predictions():
    with open(os.path.join(OUT_DIR, "cv_predictions.csv"), newline="") as f:
        return list(csv.DictReader(f))


def pooled_metrics(preds, model):
    rows = [p for p in preds if p["model"] == model]
    y = np.array([int(p["y_true"]) for p in rows])
    p = np.array([int(p["y_pred"]) for p in rows])
    n = len(y)
    correct = int((y == p).sum())
    acc, lo, hi = wilson_ci(correct, n)
    cm, prf = per_class_prf(y, p)
    f1m = float(np.mean([r[2] for r in prf]))
    return {
        "n": n, "acc": acc, "ci": (lo, hi), "macro_f1": f1m,
        "within1": within_one_accuracy(y, p), "mae": ordinal_mae(y, p),
        "cm": cm, "prf": prf,
        "err_frac": {d: float((np.abs(y - p)[np.abs(y - p) > 0] == d).mean())
                     for d in (1, 2, 3) if (np.abs(y - p) > 0).any()},
        "n_err": n - correct,
    }


def plot_pooled_cm(cm, model, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(MODEL_LABELS[model])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    preds = load_predictions()
    with open(os.path.join(ROOT, "revision", "runs", "cv_summary.json")) as f:
        fold_summary = json.load(f)

    table_rows = []
    pooled = {}
    for model in MODELS:
        m = pooled_metrics(preds, model)
        pooled[model] = m
        plot_pooled_cm(m["cm"], model, os.path.join(OUT_DIR, f"pooled_confusion_{model}.png"))
        rs = [r for r in fold_summary if r["model"] == model]
        accs = np.array([r["acc"] for r in rs])
        f1s = np.array([r["macro_f1"] for r in rs])
        w1s = np.array([r["within1_acc"] for r in rs])
        maes = np.array([r["mae"] for r in rs])
        table_rows.append({
            "model": MODEL_LABELS[model],
            "acc_pooled": f"{m['acc']*100:.2f}",
            "acc_ci": f"[{m['ci'][0]*100:.1f}, {m['ci'][1]*100:.1f}]",
            "acc_cv": f"{accs.mean()*100:.2f} ± {accs.std()*100:.2f}",
            "macro_f1": f"{m['macro_f1']:.3f}",
            "f1_cv": f"{f1s.mean():.3f} ± {f1s.std():.3f}",
            "within1": f"{m['within1']*100:.2f}",
            "within1_cv": f"{w1s.mean()*100:.2f} ± {w1s.std()*100:.2f}",
            "mae": f"{m['mae']:.3f}",
            "mae_cv": f"{maes.mean():.3f} ± {maes.std():.3f}",
        })

    with open(os.path.join(OUT_DIR, "cv_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    # markdown summary table
    lines = []
    lines.append("# 5-fold identity-grouped cross-validation results\n")
    lines.append("Every identity is predicted exactly once across folds (pooled columns); ")
    lines.append("± values are mean ± std across the 5 folds.\n")
    header = ("| Model | Acc (pooled) | 95% CI | Acc (CV) | Macro-F1 | F1 (CV) "
              "| Within-1 stage | Within-1 (CV) | MAE | MAE (CV) |")
    sep = "|---" * 10 + "|"
    lines += [header, sep]
    for r in table_rows:
        lines.append("| {model} | {acc_pooled}% | {acc_ci} | {acc_cv}% | {macro_f1} "
                     "| {f1_cv} | {within1}% | {within1_cv}% | {mae} | {mae_cv} |".format(**r))
    lines.append("")
    for model in MODELS:
        m = pooled[model]
        lines.append(f"**{MODEL_LABELS[model]}** — error structure: "
                     f"{m['n_err']} errors total; "
                     + "; ".join(f"|Δ|={d}: {frac*100:.1f}%"
                                 for d, frac in m["err_frac"].items()))
    lines.append("")
    with open(os.path.join(OUT_DIR, "cv_summary_table.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    # per-class pooled table
    plines = ["\n# Pooled per-class precision / recall / F1\n"]
    plines.append("| Model | Class | Precision | Recall | F1 | Support |")
    plines.append("|---|---|---|---|---|---|")
    for model in MODELS:
        for i, (p, r, f1, sup) in enumerate(pooled[model]["prf"]):
            plines.append(f"| {MODEL_LABELS[model]} | {CLASSES[i]} "
                          f"| {p:.3f} | {r:.3f} | {f1:.3f} | {sup} |")
    with open(os.path.join(OUT_DIR, "cv_per_class.md"), "w") as f:
        f.write("\n".join(plines))
    print("\n".join(plines))


if __name__ == "__main__":
    main()
