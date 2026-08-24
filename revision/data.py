"""Corrected data pipeline: identity-level stratified split + train-only augmentation.

Fixes the leakage identified in the review:
  1. Splits are performed at the IDENTITY level (patient / same-person session),
     so no image of the same person ever appears in more than one partition.
  2. Augmentation is applied ONLY to training identities, after the split.

Outputs:
  outputs/splits.csv               - filename, identity_id, label, split
  <AUG_BASE>/train_aug/*.jpg       - augmented training images (5 variants per original),
                                     where AUG_BASE defaults to the local temp dir
                                     (override with the MPB_AUG_DIR environment variable)

Usage:  python revision/data.py
"""
import os
import csv
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AUG_BASE, TRAIN_AUG_DIR, augment_identities

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "outputs", "manifest.csv")
SPLITS_CSV = os.path.join(ROOT, "outputs", "splits.csv")
OG_DIR = os.path.join(ROOT, "ogdata")

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20


def load_manifest():
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def stratified_identity_split(rows):
    """Assign each identity to train/val/test, stratified by identity-level label.

    Identity label = mode of its image labels (handles the one cross-class
    identity, whose images stay together in a single partition).
    """
    rng = random.Random(SEED)

    identities = {}
    for r in rows:
        identities.setdefault(r["identity_id"], []).append(r)

    by_class = {}
    for ident, members in identities.items():
        labels = [m["label"] for m in members]
        mode_label = max(set(labels), key=labels.count)
        by_class.setdefault(mode_label, []).append(ident)

    split_of = {}
    for cls in sorted(by_class):
        idents = sorted(by_class[cls])
        rng.shuffle(idents)
        n = len(idents)
        n_train = round(TRAIN_FRAC * n)
        n_test = max(1, round(TEST_FRAC * n))
        n_val = max(1, n - n_train - n_test)
        # adjust rounding so counts sum to n
        while n_train + n_val + n_test > n:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                n_train -= 1
        split_of.update({ident: "train" for ident in idents[:n_train]})
        split_of.update({ident: "val" for ident in idents[n_train:n_train + n_val]})
        split_of.update({ident: "test" for ident in idents[n_train + n_val:]})
    return split_of, identities


def verify_disjoint(split_of):
    """Hard guarantee: no identity in more than one split."""
    seen = {}
    for ident, split in split_of.items():
        if ident in seen:
            raise AssertionError(f"identity {ident} assigned twice")
        seen[ident] = split
    return True


def main():
    rows = load_manifest()
    split_of, identities = stratified_identity_split(rows)
    verify_disjoint(split_of)

    with open(SPLITS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "identity_id", "label", "split"])
        for r in rows:
            w.writerow([r["filename"], r["identity_id"], r["label"], split_of[r["identity_id"]]])

    n_aug = augment_identities(identities, split_of, TRAIN_AUG_DIR)

    # report
    print(f"Identities: {len(identities)}  |  train-only augmentation: {n_aug} images\n")
    print(f"{'split':<6} {'identities':>10} {'images':>7}  per-class images (1/2/3/4)")
    dist = {}
    for r in rows:
        s = split_of[r["identity_id"]]
        dist.setdefault(s, [0, 0, 0, [], 0])
        dist[s][0] += 1
        dist[s][4] += 1
        dist[s][3].append(r["label"])
    idents_per_split = {}
    for ident, s in split_of.items():
        idents_per_split[s] = idents_per_split.get(s, 0) + 1
    for s in ("train", "val", "test"):
        labels = dist[s][3]
        per = [labels.count(c) for c in (1, 2, 3, 4)]
        print(f"{s:<6} {idents_per_split[s]:>10} {dist[s][4]:>7}  {per}")

    # leak check across image files
    files_per_split = {}
    for r in rows:
        files_per_split.setdefault(split_of[r["identity_id"]], set()).add(r["filename"])
    assert not (files_per_split["train"] & files_per_split["val"])
    assert not (files_per_split["train"] & files_per_split["test"])
    assert not (files_per_split["val"] & files_per_split["test"])
    print("\nLeak check passed: identities and images are disjoint across splits.")
    print(f"Splits written to {os.path.relpath(SPLITS_CSV, ROOT)}")


if __name__ == "__main__":
    main()
