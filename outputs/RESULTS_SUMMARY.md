# Results Summary — Revision of MPB Classification (MDPI Reviewer Response)

This file consolidates the corrected evaluation that addresses every point raised by the
reviewer. All scripts live under `revision/`; reproducible commands are listed at the end.

---

## 0. Headline result

The originally reported **97.42% test accuracy was an artifact of patient-level data
leakage**. Under a leakage-free, identity-grouped, 5-fold cross-validation protocol, the
same custom CNN (FEv4) achieves **71.5 ± 5.2% exact accuracy**, with **within-one-stage
(Norwood) accuracy of 96.9 ± 2.7%** and **ordinal MAE of 0.32 stages**. Lightweight
ImageNet-pretrained backbones perform comparably; **EfficientNet-B0 is the strongest
(76.3 ± 3.1% exact, 100% within-one-stage, MAE 0.24)**.

---

## 1. Reviewer point → evidence mapping

### 1.1 "Out of the number of patients and images, distribution across the four classes"

**Evidence:** `outputs/audit_report.md`, `outputs/manifest.csv`

- **126 unique identities** from 130 original photographs (4 files were close-session
  duplicates of the same person and were merged; see 1.2).
- Class distribution (identities): **Group 1: 26, Group 2: 45, Group 3: 30, Group 4: 29**
  (derived from the network). Data reflects the Norwood Hamilton scale with each image
  given a numeric stage 1–4.
- Every one of the 130 original photos is one image **per identity** ("group{stage}_{id}.ext").
  Five offline augmentations (brightness, brightness/flip, contrast, sharpen — saturation
  dropped because it is a no-op after grayscale conversion) expand only the **training**
  images.

### 1.2 Is the ground truth of the Norwood–Hamilton one dermatologist?

**Honest disclosure (limitation):** Images were collected from dermatology websites where
labels were provided by the **source** (each photo was published with its Norwood stage,
often as before/after treatment pairs). **Labels were not independently re-assessed by a
board of dermatologists in this study.** This is a methodological limitation acknowledged
in the revised manuscript. The `identity` audit further revealed that some "before/after"
pairs of the **same person** appear under different filenames and could even straddle
classes; these were merged into single identities (kept within one partition) to prevent
the model from memorizing subject-level background/shirt cues.

### 1.3 "Split should be performed at patient level rather than at image level"

**This was the source of the inflation.** The original pipeline (`FEv4_train.py`,
`squeezenet.py`) loaded the augmented dataset and performed `train_test_split` at the
**image level**, so augmented near-duplicates of the same subject appeared in both
training and test sets. `metrics.py` used a *different* (stratified) re-split, so its
ROC/confusion matrices were not even from the same held-out set as training.

**Fix (implemented):** `revision/data.py` splits at the **identity level**
(`GroupShuffleSplit` style, stratified by identity class, 60/20/20) and applies
augmentation **only to the training partition**. Leak checks assert that no identity or
filename spans more than one partition. For robustness we report the **5-fold grouped
cross-validation** in `revision/cv.py`, where each subject is predicted exactly once.

### 1.4 "Augmentation applied only after the training set' has been separated"

**Fix:** Offline augmentation is generated from **training identities only**
(`common.augment_identities`). Validation and test images are always the pristine
originals.

### 1.5 "Please present the confusion matrix and class-level metrics, rather than accuracy alone"

**Both holdout and pooled cross-validation** results are provided:
- `outputs/cv_metrics.csv`, `outputs/cv_summary_table.md`
- `outputs/pooled_confusion_{model}.png` for each of the 4 models
- `outputs/holdout_confusion_matrix.png` + `outputs/holdout_metrics.md`
- per-class precision/recall/F1 (+ support) in `outputs/cv_per_class.md`

### 1.6 "Whether errors occur between adjacent severity levels or between distant categories"

**Evidence:** Error-distance analysis (pooled CV):
- FEv4: **89.2% of errors are adjacent (|Δ|=1)**, 10.8% are |Δ|=2, **0% |Δ|=3**
- SqueezeNet: 93.9% adjacent, 6.1% |Δ|=2, 0% distant
- MobileNetV3: 90.0% adjacent, 7.5% |Δ|=2, 2.5% |Δ|=3
- **EfficientNet-B0: 100% adjacent, 0% distant**

Successful classes rarely confused with distant categories. Group 1 (little/no balding) is
the most ambiguous class for every model (lowest F1), consistent with its rear/framing
variation and small support (26).

### 1.7 Generalization across patients/cameras/populations, independent external test cohort

- **Patient-level grouped CV** is the primary generalization claim (mean ± STD over
  folds, each subject seen exactly once). Equivalent results (72.4% exact, 95% CI
  54–85%) are replicated on an independent 60/20/20 holdout.
- **External cohort:** no fully independent labeled dataset was available. We provide a
  **qualitative** demonstration on real out-of-dataset scalp photographs
  (`revision/qualitative.py`), clearly labeled as anecdotal, not a statistical test.
  Cross-site, cross-camera, and population diversity beyond the current collection are
  explicitly acknowledged as limitations and future work.

### 1.8 "Comparisons with lightweight baselines such as MobileNet or EfficientNet"

**Done under identical protocol** (`revision/cv.py`): MobileNetV3-Small, EfficientNet-B0,
and SqueezeNet 1.1 (all torchvision, ImageNet weights, batch=16, 20 epochs, cosine
schedule). Table from `outputs/cv_summary_table.md`:

| Model | Acc (CV) | macro-F1 | within-1 stage | MAE |
|---|---|---|---|---|
| Custom CNN (FEv4) | 71.5 ± 5.2 | 70.3 ± 5.8 | 96.9 ± 2.7 | 0.32 |
| SqueezeNet 1.1 | 75.3 ± 9.5 | 75.0 ± 9.1 | 98.6 ± 2.9 | 0.26 |
| MobileNetV3-Small | 69.1 ± 5.1 | 61.1 ± 9.0 | 96.8 ± 1.6 | 0.35 |
| **EfficientNet-B0** | **76.3 ± 3.1** | **75.3 ± 3.6** | **100.0 ± 0.0** | **0.24** |

### 1.9 Deployment efficiency (params, FLOPs, latency, model size)

To clarify the trade-off between accuracy and deployment cost we measured the standard
efficiency metrics (single forward pass at 256×256; CPU = Apple MPS/M-series native CPU;
GPU = Apple M-series MPS; fp32 sizes). Generated by `revision/deploy_metrics.py`
(→ `outputs/deployment_metrics.csv`):

| Model | Params | FLOPs | fp32 size | Inference | Inference |
|---|---|---|---|---|---|
| | (M) | (M) | (MB) | CPU (ms) | MPS (ms) |
|---|---|---|---|---|---|
| **Custom CNN (FEv4)** | 16.87 | **345** | 67.5 | **4.0** | ~0.5–1.3 |
| SqueezeNet 1.1 | 0.72 | 695 | **2.9** | 7.3 | ~1.1 |
| MobileNetV3-Small | 1.52 | 4,813 | 6.1 | 37.3 | ~5.2 |
| EfficientNet-B0 | 4.01 | 40,048 | 16.1 | 138.8 | ~8.0 |

**Reading (honest):**
- **Compute / latency: FEv4 is the clear winner.** It needs only **345 MFLOPs — about
  116× less compute than EfficientNet-B0 and 14× less than MobileNetV3-Small** — and has
  the **fastest CPU inference (~4 ms)** and competitive GPU inference. For a server-hosted
  web/Streamlit deployment (e.g. the deployed demo), this means near-instant, cheap,
  per-request prediction, which is why the custom CNN deploys cleanly.
- **Size / parameters / on-device: FEv4 loses.** At 67.5 MB (fp32) / 16.9 M params it is the
  **largest** model — about 23× SqueezeNet and 11× MobileNetV3 in file size. It is *not*
  a lightweight or edge/on-device model in the size sense; SqueezeNet (2.9 MB) and
  MobileNetV3 (6.1 MB) are, and would be preferred if the target is browsers or mobile edge
  devices.

**Net deployment conclusion.** The accurate characterization of the custom CNN is: a
**low-compute, low-latency, from-scratch** model that is well suited to **cloud / web**
telemedicine (low FLOPs, fast server inference, no pretrained-weight dependency), but not a
**small-footprint / on-device** model. For genuinely resource-constrained edge deployment,
SqueezeNet/MobileNetV3 are the portable options, while EfficientNet-B0 offers the best
accuracy at moderate size. This trade-off, rather than a claim of general superiority, is
what the revised manuscript reports.

---

## 2. Summary of files

- `outputs/audit_report.md`, `outputs/manifest.csv`, `outputs/duplicate_pairs.csv` — data audit
- `outputs/splits.csv` — identity-level 60/20/20 split
- `outputs/cv_summary_table.md`, `outputs/cv_metrics.csv`, `outputs/cv_per_class.md`,
  `outputs/pooled_confusion_*.png`, `revision/runs/cv_summary.json` — CV results
- `outputs/holdout_metrics.md`, `outputs/holdout_confusion_matrix.png` — holdout evaluation
- `outputs/deployment_metrics.csv` — params/FLOPs/latency/size table
- `revision/qualitative.py` — external anecdotal inference

## 3. Reproduce the numbers

```bash
# 1 env
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchvision scikit-learn pillow numpy matplotlib imagehash

# 2 audit + splits
python revision/audit.py
python revision/data.py

# 3 full cross-validation (all 4 models, 5 grouped folds)
python revision/cv.py

# 4 report + figures
python revision/report.py

# 5 holdout (single 60/20/20) + qualitative
python revision/evaluate.py
python revision/qualitative.py

# 6 deployment-efficiency metrics (params / FLOPs / latency / size)
python revision/deploy_metrics.py
```

> Note: augmentation is written to a local temp dir (AUG_BASE, overridable via the
> `MPB_AUG_DIR` env var) to avoid iCloud/OneDrive conflict copies (e.g. "file 2.jpg")
> polluting the dataset.