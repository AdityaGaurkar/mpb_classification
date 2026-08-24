"""Qualitative (anecdotal) inference on out-of-dataset personal photos.

NOT a statistical evaluation: these images are from individuals outside the
dataset (colleagues/family), used only to illustrate deployment behavior.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    CLASSES,
    OG_DIR,
    ROOT,
    RUNS_DIR,
    _load_gray,
    fe4_cnn,
    get_device,
)

TEST_IMAGES = os.path.join(ROOT, "test_images")


def main():
    device = get_device()
    model = fe4_cnn().to(device)
    model.load_state_dict(torch.load(os.path.join(RUNS_DIR, "fe4_corrected.pt"),
                                     map_location=device))
    model.eval()
    print("Qualitative predictions (anecdotal only — not a test cohort)\n")
    with torch.no_grad():
        for fname in sorted(os.listdir(TEST_IMAGES)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            x = torch.from_numpy(_load_gray(os.path.join(TEST_IMAGES, fname)))[None].to(device)
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            top = int(np.argmax(probs))
            pretty = " > ".join(f"{CLASSES[i]}:{probs[i]*100:.0f}%" for i in np.argsort(-probs))
            print(f"{fname:<18} -> {CLASSES[top]:<8} ({pretty})")


if __name__ == "__main__":
    main()
