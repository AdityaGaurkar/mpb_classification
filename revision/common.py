"""Shared utilities: datasets, models (FEv4 + pretrained baselines), training, metrics."""
import csv
import math
import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OG_DIR = os.path.join(ROOT, "ogdata")
SPLITS_CSV = os.path.join(ROOT, "outputs", "splits.csv")
RUNS_DIR = os.path.join(ROOT, "revision", "runs")
# Generated augmented images go to a local, non-synced directory: the project
# lives in iCloud-synced ~/Documents, and rapid file churn there creates
# conflict copies ("name 2.jpg") that would silently pollute training.
AUG_BASE = os.environ.get("MPB_AUG_DIR", os.path.join(tempfile.gettempdir(), "mpb_aug"))
TRAIN_AUG_DIR = os.path.join(AUG_BASE, "train_aug")
IMG_SIZE = 256
NUM_CLASSES = 4
CLASSES = ("group1", "group2", "group3", "group4")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        pass  # MPS seeding is handled by torch.manual_seed
    torch.use_deterministic_algorithms(False)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_splits():
    """Return dict split -> list of (filename, identity_id, label)."""
    out = {"train": [], "val": [], "test": []}
    with open(SPLITS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            out[r["split"]].append((r["filename"], r["identity_id"], int(r["label"])))
    return out


def _load_gray(path):
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, :, :]  # (1, H, W)


class MPBDataset(Dataset):
    """train -> augmented copies (aug_dir); val/test -> originals (ogdata/).

    channels=1 for the grayscale custom CNN; channels=3 replicates the gray
    image for ImageNet-pretrained backbones.
    """

    def __init__(self, split, splits=None, channels=1, aug_dir=None):
        self.channels = channels
        if split == "train":
            aug_dir = aug_dir or TRAIN_AUG_DIR
            id2label = {}
            for fname, _ident, label in splits["train"]:
                id2label[fname.replace(" ", "_").rsplit(".", 1)[0]] = label - 1
            self.samples = []
            for fname in sorted(os.listdir(aug_dir)):
                if not fname.lower().endswith(".jpg"):
                    continue
                stem = fname.rsplit("__", 1)[0]
                self.samples.append((os.path.join(aug_dir, fname), id2label[stem]))
        else:
            self.samples = [
                (os.path.join(OG_DIR, fname), label - 1)
                for fname, _ident, label in splits[split]
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = _load_gray(path)
        if self.channels == 3:
            arr = np.repeat(arr, 3, axis=0)
        return torch.from_numpy(arr), label


def fe4_cnn(num_classes=NUM_CLASSES):
    """Faithful PyTorch port of the Keras FEv4 model (training_scripts/FEv4_train.py).

    Conv(32,s2)-BN-ReLU -> Conv(64,s2)-BN-ReLU -> Conv(128,s2)-BN-ReLU ->
    Flatten -> Dense(128)-ReLU -> Dropout(0.5) -> Dense(num_classes).
    """
    layers = []
    ch_in, filters = 1, (32, 64, 128)
    for f in filters:
        conv = nn.Conv2d(ch_in, f, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        layers += [conv, nn.BatchNorm2d(f, momentum=0.99), nn.ReLU(inplace=True)]
        ch_in = f
    model = nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(128 * (IMG_SIZE // 8) ** 2, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )
    for m in (model[-4], model[-1]):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    return model


def cosine_lr(epoch, total_epochs=20, min_lr=1e-5, max_lr=1e-3):
    return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2


def _adapt_pretrained(model, num_classes):
    """Replace the ImageNet classification head with a 4-class head."""
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):  # MobileNetV3
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:  # EfficientNet
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def make_model(name, num_classes=NUM_CLASSES):
    if name == "fe4":
        return fe4_cnn(num_classes), 1, 1e-3
    import torchvision.models as tvm

    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        return _adapt_pretrained(m, num_classes), 3, 1e-4
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        return _adapt_pretrained(m, num_classes), 3, 1e-4
    if name == "squeezenet1_1":
        m = tvm.squeezenet1_1(weights=tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        return m, 3, 1e-4
    raise ValueError(f"unknown model: {name}")


MODELS = ("fe4", "squeezenet1_1", "mobilenet_v3_small", "efficientnet_b0")


def augment_identities(identities, split_of, out_dir):
    """Offline augmentation (original, brightness, flip, contrast, sharpen)
    for TRAIN identities only; mirrors the paper's original transforms."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    n = 0
    for ident, members in sorted(identities.items()):
        if split_of[ident] != "train":
            continue
        for m in members:
            img = Image.open(os.path.join(OG_DIR, m["filename"])).convert("RGB")
            stem = m["patient_id"].replace(" ", "_")
            variants = (
                img,
                ImageEnhance.Brightness(img).enhance(1.4),
                img.transpose(Image.FLIP_LEFT_RIGHT),
                ImageEnhance.Contrast(img).enhance(1.5),
                img.filter(ImageFilter.SHARPEN),
            )
            for i, v in enumerate(variants):
                v.save(os.path.join(out_dir, f"{stem}__{i}.jpg"), quality=95)
                n += 1
    written = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    if written != n:
        raise RuntimeError(
            f"augmented dir has {written} files, expected {n} — external interference "
            f"(sync conflict copies?) in {out_dir}"
        )
    return n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_model(model, train_loader, val_loader, device, epochs=20, patience=5, seed=42,
                max_lr=1e-3, min_lr=1e-5):
    """Replicates Keras config: Adam + cosine-annealed LR, early stopping on val loss."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=max_lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch = math.inf, None, -1
    since_best = 0
    history = []

    for epoch in range(epochs):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(epoch, epochs, min_lr=min_lr, max_lr=max_lr)
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_n += len(yb)

        model.eval()
        va_loss, va_correct, va_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                va_loss += loss_fn(logits, yb).item() * len(yb)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_n += len(yb)

        rec = {
            "epoch": epoch,
            "lr": cosine_lr(epoch, epochs, min_lr=min_lr, max_lr=max_lr),
            "train_loss": tr_loss / tr_n,
            "train_acc": tr_correct / tr_n,
            "val_loss": va_loss / va_n,
            "val_acc": va_correct / va_n,
        }
        history.append(rec)
        marker = ""
        if rec["val_loss"] < best_val:
            best_val, best_epoch, since_best = rec["val_loss"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        else:
            since_best += 1
        print(
            f"epoch {epoch:02d}  lr {rec['lr']:.2e}  "
            f"train_loss {rec['train_loss']:.4f} acc {rec['train_acc']:.3f}  "
            f"val_loss {rec['val_loss']:.4f} acc {rec['val_acc']:.3f}{marker}"
        )
        if since_best >= patience:
            print(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    return history, best_epoch


# ---------------- metrics ----------------

def confusion_matrix(y_true, y_pred, k=NUM_CLASSES):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1(y_true, y_pred, k=NUM_CLASSES):
    cm = confusion_matrix(y_true, y_pred, k)
    f1s = []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s)), np.array(f1s)


def per_class_prf(y_true, y_pred, k=NUM_CLASSES):
    cm = confusion_matrix(y_true, y_pred, k)
    rows = []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((prec, rec, f1, int(cm[c, :].sum())))
    return cm, rows


def within_one_accuracy(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def ordinal_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def error_distance_fractions(y_true, y_pred, k=NUM_CLASSES):
    d = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    errs = d[d > 0]
    if len(errs) == 0:
        return {i: 0.0 for i in range(1, k)}
    return {i: float((errs == i).mean()) for i in range(1, k)}


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, centre - half, centre + half
