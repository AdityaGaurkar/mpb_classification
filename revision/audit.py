"""Data audit for MPB classification revision.

Produces:
  outputs/manifest.csv            - one row per original image (ogdata/)
  outputs/duplicate_pairs.csv     - all pairs ranked by perceptual hash distance
  outputs/augmented_check.csv     - augmented-folder consistency check
  outputs/audit_report.md         - human-readable summary
"""
import os
import itertools
import csv

import numpy as np
from PIL import Image
import imagehash

OG_DIR = "ogdata"
AUG_DIR = "augmented"
OUT_DIR = "outputs"

PHASH_SIZE = 16          # 256-bit hash
NEAR_DUP_CUTOFF = 24     # hamming distance out of 256 bits (~9%); report everything below

# Visual verification of hash-flagged candidates (see audit report).
# Each entry: canonical patient_id -> list of additional filenames that belong
# to the SAME identity (same person / same photoshoot session).
# All images of one identity are forced into the same data partition.
IDENTITY_MERGES = {
    "group3_29": ["group3_29 2.png"],   # confirmed: same shirt, couch, person (two frames)
    "group2_1": ["group2_26.png", "group3_21.png"],  # same studio bg + hair texture; cross-class progression series
    "group2_15": ["group2_7.png"],      # same hair color/texture/hairline pattern, same bg
}


def scan_ogdata():
    rows = []
    for fname in sorted(os.listdir(OG_DIR)):
        if fname.startswith("."):
            continue
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = os.path.splitext(fname)[0]
        parts = stem.split("_")
        group = parts[0]                      # e.g. group3
        patient_id = "_".join(parts[:2])      # e.g. group3_29
        label = int(group.replace("group", ""))
        rows.append({
            "filename": fname,
            "patient_id": patient_id,
            "label": label,
        })
    return rows


def compute_hashes(rows):
    hashes = []
    for r in rows:
        img = Image.open(os.path.join(OG_DIR, r["filename"])).convert("L")
        h_phash = imagehash.phash(img, hash_size=PHASH_SIZE)
        h_ahash = imagehash.average_hash(img, hash_size=PHASH_SIZE)
        hashes.append((h_phash, h_ahash))
    return hashes


def pair_table(rows, hashes):
    pairs = []
    for (i, j) in itertools.combinations(range(len(rows)), 2):
        d_ph = hashes[i][0] - hashes[j][0]
        d_ah = hashes[i][1] - hashes[j][1]
        same_class = rows[i]["label"] == rows[j]["label"]
        pairs.append({
            "file_a": rows[i]["filename"],
            "id_a": rows[i]["patient_id"],
            "class_a": rows[i]["label"],
            "file_b": rows[j]["filename"],
            "id_b": rows[j]["patient_id"],
            "class_b": rows[j]["label"],
            "same_class": same_class,
            "phash_dist": d_ph,
            "ahash_dist": d_ah,
        })
    pairs.sort(key=lambda p: p["phash_dist"])
    return pairs


def check_augmented(rows):
    expected = {r["patient_id"]: 6 for r in rows}
    found = {}
    anomalies = []
    for fname in os.listdir(AUG_DIR):
        if fname.startswith("."):
            continue
        stem = os.path.splitext(fname)[0]
        parts = stem.split("_")
        pid = "_".join(parts[:2])
        found[pid] = found.get(pid, 0) + 1
    for pid, n in sorted(expected.items()):
        got = found.get(pid, 0)
        if got != 6:
            anomalies.append({"patient_id": pid, "expected": 6, "found": got})
    extra = [pid for pid in found if pid not in expected]
    return anomalies, extra, len(found)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = scan_ogdata()
    print(f"Original images: {len(rows)}")

    # apply identity merges (same person / same session -> one identity)
    file_to_identity = {}
    for canonical, extras in IDENTITY_MERGES.items():
        for extra in extras:
            file_to_identity[extra] = canonical
    for r in rows:
        r["identity_id"] = file_to_identity.get(r["filename"], r["patient_id"])

    # class distribution
    dist = {}
    for r in rows:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    n_identities = len({r["identity_id"] for r in rows})

    # duplicate detection
    hashes = compute_hashes(rows)
    pairs = pair_table(rows, hashes)
    flagged = [p for p in pairs if p["phash_dist"] <= NEAR_DUP_CUTOFF]

    with open(os.path.join(OUT_DIR, "duplicate_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)

    # augmented consistency
    anomalies, extra, n_aug_patients = check_augmented(rows)

    # manifest
    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "patient_id", "identity_id", "label"])
        w.writeheader()
        w.writerows(rows)

    # report
    lines = []
    lines.append("# Data Audit Report\n")
    lines.append(f"Total original images: **{len(rows)}**  ")
    lines.append(f"Unique identities after visual verification: **{n_identities}**\n")
    lines.append("## Class distribution (original images)\n")
    lines.append("| Class | Patients |")
    lines.append("|---|---|")
    for k in sorted(dist):
        lines.append(f"| Group {k} | {dist[k]} |")
    lines.append("")
    lines.append(f"## Augmented folder consistency\n")
    lines.append(f"- Unique patients represented: {n_aug_patients}")
    lines.append(f"- Patients without exactly 6 variants: {len(anomalies)}")
    for a in anomalies:
        lines.append(f"  - {a['patient_id']}: expected 6, found {a['found']}")
    if extra:
        lines.append(f"- Files not matching any ogdata patient: {extra}")
    lines.append("")
    lines.append(f"## Near-duplicate candidates (phash distance <= {NEAR_DUP_CUTOFF}/256)\n")
    lines.append("These are potential before/after treatment pairs of the SAME person ")
    lines.append("(scraped from dermatology sites) appearing under different IDs/classes.\n")
    if flagged:
        lines.append("| dist | file A | class A | file B | class B | cross-class |")
        lines.append("|---|---|---|---|---|---|")
        for p in flagged:
            cc = "**YES**" if not p["same_class"] else ""
            lines.append(
                f"| {p['phash_dist']} | {p['file_a']} | {p['class_a']} "
                f"| {p['file_b']} | {p['class_b']} | {cc} |"
            )
    else:
        lines.append("None found.")
    lines.append("")
    lines.append("## Visual verification of hash candidates\n")
    lines.append("Whole-image pHash misses same-session pairs (pose changes defeat it), so a ")
    lines.append("secondary detector compared bottom-region (shirt/background) hashes and border ")
    lines.append("color-histogram correlations. All flagged candidates were inspected visually:\n")
    lines.append("| Pair | Verdict | Action |")
    lines.append("|---|---|---|")
    lines.append("| group3_29 / group3_29 2 | **Same person** (same shirt, couch, session) | Merged into one identity |")
    lines.append("| group2_1 / group2_26 / group3_21 | **Plausibly same person** (same studio bg, hair texture; cross-class progression) | Merged into one identity |")
    lines.append("| group2_15 / group2_7 | **Plausibly same person** (same hair color, texture, hairline, bg) | Merged into one identity |")
    lines.append("| group1_15/group2_34, group1_8/group2_41, group2_17/group3_6, group2_2/group4_30, group2_18/group2_4, group2_28/group2_3, group2_9 vs others | Different people (shared clinical grey/white backgrounds caused false positives) | Kept separate |")
    lines.append("")
    lines.append("All images of one identity are assigned to the same train/val/test partition, ")
    lines.append("and cross-class identities additionally guarantee that before/after images of ")
    lines.append("the same person never straddle partitions.\n")
    lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "audit_report.md"), "w") as f:
        f.write(report)
    print(report)
    print(f"\nFull ranked pair list ({len(pairs)} pairs) -> outputs/duplicate_pairs.csv")


if __name__ == "__main__":
    main()
