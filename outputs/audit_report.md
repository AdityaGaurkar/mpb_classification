# Data Audit Report

Total original images: **130**  
Unique identities after visual verification: **126**

## Class distribution (original images)

| Class | Patients |
|---|---|
| Group 1 | 26 |
| Group 2 | 45 |
| Group 3 | 30 |
| Group 4 | 29 |

## Augmented folder consistency

- Unique patients represented: 129
- Patients without exactly 6 variants: 1
  - group3_29 2: expected 6, found 0

## Near-duplicate candidates (phash distance <= 24/256)

These are potential before/after treatment pairs of the SAME person 
(scraped from dermatology sites) appearing under different IDs/classes.

None found.

## Visual verification of hash candidates

Whole-image pHash misses same-session pairs (pose changes defeat it), so a 
secondary detector compared bottom-region (shirt/background) hashes and border 
color-histogram correlations. All flagged candidates were inspected visually:

| Pair | Verdict | Action |
|---|---|---|
| group3_29 / group3_29 2 | **Same person** (same shirt, couch, session) | Merged into one identity |
| group2_1 / group2_26 / group3_21 | **Plausibly same person** (same studio bg, hair texture; cross-class progression) | Merged into one identity |
| group2_15 / group2_7 | **Plausibly same person** (same hair color, texture, hairline, bg) | Merged into one identity |
| group1_15/group2_34, group1_8/group2_41, group2_17/group3_6, group2_2/group4_30, group2_18/group2_4, group2_28/group2_3, group2_9 vs others | Different people (shared clinical grey/white backgrounds caused false positives) | Kept separate |

All images of one identity are assigned to the same train/val/test partition, 
and cross-class identities additionally guarantee that before/after images of 
the same person never straddle partitions.

