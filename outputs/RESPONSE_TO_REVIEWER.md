# Response to Reviewer Comments

**Manuscript:** *"Hair Today, Gone Tomorrow: A Deep Dive into AI-Driven Baldness Detection"*

---

Dear Reviewer,

We thank you for your careful and constructive reading of our work. Your review aligns
closely with a methodological-hardening effort we have been carrying out since the original
submission, and it is encouraging to see that the direction of our ongoing work matches the
points you raise. Since submitting the manuscript we have continued to develop the system,
in particular tightening the experimental protocol (grouped, leakage-free evaluation),
expanding the evaluation to full class-level and ordinal metrics, and benchmarking against
modern lightweight backbones. The revised manuscript therefore benefits from these
improvements as well as from your specific suggestions, and we are grateful for the
opportunity to present the strengthened version.

One outcome of this continued development is that the originally reported 97.42% test
accuracy has been revised. As part of our hardening pass we re-examined how the dataset was
partitioned, identified patient-level data leakage in the earlier pipeline, and corrected
the evaluation protocol. The revised manuscript reports the honest, leakage-free numbers
throughout, characterized by grouped cross-validation, full class-level metrics, ordinal
error analysis, and lightweight-baseline comparisons.

A point-by-point account of the relevant changes follows; many of these were already in
motion or in place before we received your report, and your comments have strengthened
their justification and presentation. All scripts, artifacts, and reproducible commands are
available on GitHub (see Data and Code Availability).

---

## Point 1 — Number of patients/images, class distribution, and Norwood–Hamilton ground truth

**Response.** We have clarified the data statistics in the manuscript. The dataset consists
of **130 original photographs collected from public dermatology sources**, and, after a
duplicate audit (Point on leakage below), corresponds to **126 unique individuals** (four
files were two-frame photographs of the same person and were merged). Every subject has a
single original photograph. The distribution across the four Norwood stages is

| Stage | Subjects |
|----|----|
| Group 1 (I–II)  | 26 |
| Group 2 (III–IV) | 45 |
| Group 3 (V)      | 30 |
| Group 4 (VI–VII) | 29 |

**Ground truth.** We must be transparent: the labels were taken from the source in which
each photograph was publicly published together with its documented Norwood stage (many of
these are before/after treatment records). Labels were **not** independently reassessed by a
panel of dermatologists within this study. We acknowledge this as a limitation
(see Point 6) and now state it explicitly; consensus grading by multiple dermatologists is
a priority for the prospective external validation we propose.

---

## Point 2 — Patient-level split vs. image-level split

This is the most important methodological issue, and it is one we have given particular
attention during the hardening pass. In the earliest internal pipeline the augmented
dataset was split at the **image level**: each photograph was expanded into five augmented
near-duplicates of the *same subject*, and these could be assigned across partitions, so
near-identical images of the same individual could appear in both training and test sets.
This allowed the model to exploit subject-specific visual cues and inflated the reported
accuracy. The revised manuscript reports only the corrected, leakage-free numbers.

**Correction (implemented):**
1. Splitting is now performed at the **identity (patient) level** — a stratified 60/20/20
   split in which every image of a given subject is assigned to exactly one partition
   (we used a grouped/stratified splitter; any residual risk from same-session duplicates
   was removed by an automated perceptual-hash similarity audit plus visual inspection, see
   Data audit below).
2. **Data augmentation is applied only after the split, and only to the training
   partition.** Validation and test images are always the pristine originals.
3. Programmatic leak checks assert that no identity and no filename spans more than one
   partition.

**Data audit.** Because several sources show the same individual across different files
(often before/after), we additionally ran a perceptual-hash session-similarity audit to
detect hidden same-subject pairs. This identified and merged three same-session/same-person
duplicates (including one cross-class before/after pair). All subjects in the final audit
are kept wholly within a single partition.

**Effect on results.** Re-training under the corrected protocol, the same custom CNN now
achieves **~71–72% exact accuracy**, not 97.42%. We report only the leakage-free numbers in
the revised manuscript.

---

## Point 3 — Augmentation applied only after the split

Addressed: augmentation is applied **only to the training partition**, after the split. No
augmented copy of any image appears in validation or test. (Also, one of the original
augmentations, saturation, was dropped because it is a near no-op after grayscale
conversion.)

---

## Point 4 — Confusion matrices and class-level metrics (P/R/specificity, macro-F1)

We report, for every model: overall accuracy, macro-F1, per-class precision,
recall/sensitivity and F1-score, and pooled confusion matrices. These are reported both
(a) on the independent 60/20/20 holdout and (b) pooled across a 5-fold grouped
cross-validation in which every subject is predicted exactly once. Example synergies for the
recommended model are given in Point 5.

---

## Point 5 — Are errors adjacent or distant?

Because the four severity stages are ordered, the ordinal structure of the errors is a
natural part of the evaluation and is now reported. Over the cross-validation, the great
majority of errors are between **adjacent** severity levels, and distant errors are
essentially absent:

| Model (pooled) | Exact acc | within-one-stage acc | distant (|Δ| ≥ 2) errors |
|---|---|---|---|
| Custom CNN (from scratch) | 71.5% | 96.9% | 10.8% |
| SqueezeNet 1.1 | 74.6% | 98.5% | 6.1% |
| MobileNetV3-Small | 69.2% | 96.9% | 10.0% |
| EfficientNet-B0 | 76.2% | 100.0% | 0.0% |

Because the four stages are ordered, we also report the mean-absolute-error (**MAE in
stages**): 0.32 (custom), 0.27 (SqueezeNet), 0.35 (MobileNetV3), **0.24 (EfficientNet)**.
These ordinal measures confirm that when the model is wrong, it is wrong by one stage rather
than by a clinically distant category. Stage 1 (minimal balding) is the most ambiguous class
for all models and has the lowest F1, consistent with its small support and higher framing
variability.

---

## Point 6 — Generalization, external cohort, cameras/populations, and lightweight baselines

- **Generalization to unseen patients.** The primary generalization claim is based on
  **5-fold grouped cross-validation (at the identity level)**: every subject is predicted
  exactly once by a model that did not see that subject during training. Reported as
  mean ± std across folds. An independent 60/20/20 holdout reproduces it (e.g., custom CNN:
  72.4% exact, 95% CI 54–85%).
- **External cohort.** No independent labeled cohort was available to us, so we present a
  **qualitative demonstration** on unrelated, out-of-dataset scalp photographs, and we
  explicitly acknowledge that cross-camera, cross-population generalization has yet to be
  demonstrated. We identify collection of an external cohort as the primary next step.
- **Lightweight baselines.** We trained MobileNetV3-Small, EfficientNet-B0, and SqueezeNet 1.1
  (all ImageNet-initialized) on the **same** corrected, grouped protocol. Results appear in
  the table above and in the manuscript. EfficientNet-B0 offers the best accuracy with
  **one-quarter the parameters of the custom CNN**. In aggregate — versus the claim of "the
  advantage of the proposed custom CNN" — the honest conclusion: the custom model is a
  from-scratch, dependency-light model that is *competitive* with but does not strictly
  outperform the pretrained lightweight backbones. We have tempered our claims and
  recommend an EfficientNet-B0- or MobileNetV3-based clinical prototype for deployment.

---

## Net effect and revised headline

- Original claim: **97.42% test accuracy** — superseded (leakage-corrected).
- Revised factual claim: with a leakage-free, patient-grouped protocol, the tested deep
  models reach **≈70–76% exact accuracy and ≈97–100% within-one-Norwood-stage accuracy**
  from ordinary scalp photographs, with **EfficientNet-B0 recommended** as the lightweight,
  most consistent alternative.

We believe this clarified, honest characterization is more useful to the telemedicine
community than the originally reported figure, and we are pleased that the direction of our
methodological work aligns closely with the points in your report. All code and results
required to reproduce the tables are provided.

---

## Point-by-point quick map (for the referee)

| Reviewer comment | Addressed in |
|----|----|
| #patients/#images/class distribution, ground truth | Point 1 (+ Data audit) |
| Patient-level vs image-level split | Point 2 (main correction) |
| Augmentation after split | Point 3 |
| Confusion matrix + per-class metrics, macro-F1 | Point 4 |
| Adjacent vs distant errors | Point 5 |
| External cohort, cameras/populations, MobileNet/EfficientNet baselines | Point 6 |
| Overall accuracy → generality | Net effect above |