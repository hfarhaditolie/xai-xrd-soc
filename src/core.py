import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import fabio
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Data")
RESULTS_DIR = os.path.join(ROOT, "results")

FRESH = {
    "images": os.path.join(DATA_DIR, "fresh", "2D"),
    "patterns": os.path.join(DATA_DIR, "fresh", "1D"),
    "metadata": os.path.join(DATA_DIR, "fresh", "metadata.xlsx"),
}
AGED = {
    "images": os.path.join(DATA_DIR, "aged", "2D"),
    "patterns": None,
    "metadata": os.path.join(DATA_DIR, "aged", "metadata.xlsx"),
}

TARGET_COLUMN = "SoC_Coulomb"
FILE_COLUMN = "File"
RATE_COLUMN = "Cycling_Rate"
CYCLE_COLUMN = "Cycle_Number"
IMG_SIZE = (224, 224)

TTH_MIN = 16.233163
TTH_MAX = 42.184300

NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EVAL_BATCH_SIZE = 64

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
torch.backends.cudnn.benchmark = True

AMP_DTYPE = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16)

BACKBONES = [
    "ResNet18", "ResNet50", "ResNet101",
    "EfficientNetB0", "EfficientNetB1", "EfficientNetB2",
    "DenseNet121", "DenseNet169", "DenseNet201",
    "MobileNetV2", "MobileNetV3Small", "MobileNetV3Large",
    "VGG16", "VGG19",
]

SPLIT_SCHEMES = ["Leave-one-cycle-out", "Grouped K-fold by cycle",
                 "Leave-one-C-rate-out", "Random K-fold"]


def frame_key(name):
    base = os.path.splitext(os.path.basename(str(name)))[0]
    return re.sub(r"_azimavg$", "", base, flags=re.IGNORECASE)


def load_dataset(cell="fresh"):
    spec = FRESH if cell == "fresh" else AGED
    df = pd.read_excel(spec["metadata"]).dropna(subset=[TARGET_COLUMN]).copy()
    df["_key"] = [frame_key(f) for f in df[FILE_COLUMN]]
    lookup = df.set_index("_key")

    files = sorted(f for f in os.listdir(spec["images"]) if f.endswith((".edf", ".tif")))
    matched = [(f, frame_key(f)) for f in files if frame_key(f) in lookup.index]
    if not matched:
        raise ValueError(f"no frames in {spec['images']} matched {spec['metadata']}")

    images, records = [], []
    for fname, key in matched:
        raw = fabio.open(os.path.join(spec["images"], fname)).data
        norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        images.append(cv2.resize(norm, (IMG_SIZE[1], IMG_SIZE[0]),
                                 interpolation=cv2.INTER_AREA))
        row = lookup.loc[key]
        records.append({
            "filename": fname,
            "target": float(row[TARGET_COLUMN]),
            "rate": str(row[RATE_COLUMN]),
            "cycle": int(row[CYCLE_COLUMN]),
        })

    meta = pd.DataFrame(records)
    meta["cycle"] = meta["cycle"] - meta["cycle"].min() + 1
    return images, meta


def read_pattern(path):
    xs, ys = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except (ValueError, IndexError):
                continue
    return np.asarray(xs), np.asarray(ys)


def soc_intervals(meta, n_bins=4):
    rel = meta["target"].values / meta["target"].max() * 100.0
    edges = np.linspace(0, 100, n_bins + 1)
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}%" for i in range(n_bins)]
    return pd.cut(rel, bins=edges, labels=labels,
                  include_lowest=True).astype(str), labels


def charge_discharge_labels(meta, window=5):
    branch = np.zeros(len(meta), dtype=np.int64)
    for _, group in meta.groupby("cycle"):
        soc = group["target"].values
        smooth = pd.Series(soc).rolling(window, center=True, min_periods=1).mean().values
        grad = np.gradient(smooth)
        lab = np.sign(grad)
        lab[np.abs(grad) < 0.02 * np.ptp(smooth) / len(smooth) * 3] = 0
        for i in range(len(lab)):
            if lab[i] == 0 and i > 0:
                lab[i] = lab[i - 1]
        for i in range(len(lab) - 1, -1, -1):
            if lab[i] == 0 and i < len(lab) - 1:
                lab[i] = lab[i + 1]
        lab[lab == 0] = 1
        branch[group.index.to_numpy()] = (lab > 0).astype(np.int64)
    return branch


def make_splits(meta, scheme="Leave-one-cycle-out", n_splits=8, seed=42):
    idx = np.arange(len(meta))
    if scheme == "Leave-one-cycle-out":
        groups = meta["cycle"].values
        return [(f"Cycle {groups[te][0]} ({meta['rate'].values[te][0]})", tr, te)
                for tr, te in LeaveOneGroupOut().split(idx, groups=groups)]
    if scheme == "Grouped K-fold by cycle":
        groups = meta["cycle"].values
        k = min(n_splits, meta["cycle"].nunique())
        return [(f"Fold {i} (cycles {', '.join(map(str, sorted(set(groups[te]))))})", tr, te)
                for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(idx, groups=groups), 1)]
    if scheme == "Leave-one-C-rate-out":
        groups = meta["rate"].values
        return [(f"Held-out rate {groups[te][0]}", tr, te)
                for tr, te in LeaveOneGroupOut().split(idx, groups=groups)]
    if scheme == "Random K-fold":
        return [(f"Fold {i}", tr, te) for i, (tr, te) in
                enumerate(KFold(n_splits=n_splits, shuffle=True,
                                random_state=seed).split(idx), 1)]
    raise ValueError(scheme)


class FrameDataset(Dataset):
    def __init__(self, images, targets, filenames):
        self.images = images
        self.targets = np.asarray(targets, dtype=np.float32)
        self.filenames = filenames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).unsqueeze(0).float().div_(255.0)
        return x, torch.tensor(self.targets[i]), self.filenames[i]


class SequenceDataset(Dataset):
    def __init__(self, images, targets, filenames, windows):
        self.images = images
        self.targets = np.asarray(targets, dtype=np.float32)
        self.filenames = filenames
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        frames = [torch.from_numpy(self.images[j]).unsqueeze(0).float().div_(255.0)
                  for j in self.windows[i]]
        return torch.stack(frames), torch.tensor(self.targets[i]), self.filenames[i]


class BranchDataset(Dataset):
    def __init__(self, images, targets, branch, filenames):
        self.images = images
        self.targets = np.asarray(targets, dtype=np.float32)
        self.branch = np.asarray(branch, dtype=np.int64)
        self.filenames = filenames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).unsqueeze(0).float().div_(255.0)
        return x, torch.tensor(self.targets[i]), torch.tensor(self.branch[i])


def build_windows(meta, seq_len):
    windows = np.zeros((len(meta), seq_len), dtype=np.int64)
    for _, group in meta.groupby("cycle", sort=True):
        idx = group.index.to_numpy()
        if not np.all(np.diff(idx) == 1):
            raise ValueError("frames within a cycle are not contiguous")
        for pos, i in enumerate(idx):
            w = idx[max(0, pos - seq_len + 1): pos + 1]
            if len(w) < seq_len:
                w = np.concatenate([np.repeat(w[0], seq_len - len(w)), w])
            windows[i] = w
    return windows


def _grayscale_conv(conv):
    return nn.Conv2d(1, conv.out_channels, kernel_size=conv.kernel_size,
                     stride=conv.stride, padding=conv.padding,
                     bias=conv.bias is not None)


def _head(dim, hidden=(512, 128), dropout=(0.3, 0.2), out=1):
    return nn.Sequential(
        nn.Linear(dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout[0]),
        nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout[1]),
        nn.Linear(hidden[1], out),
    )


_SPECS = {
    "ResNet18": (models.resnet18, models.ResNet18_Weights, 512, "resnet"),
    "ResNet50": (models.resnet50, models.ResNet50_Weights, 2048, "resnet"),
    "ResNet101": (models.resnet101, models.ResNet101_Weights, 2048, "resnet"),
    "EfficientNetB0": (models.efficientnet_b0, models.EfficientNet_B0_Weights, 1280, "features"),
    "EfficientNetB1": (models.efficientnet_b1, models.EfficientNet_B1_Weights, 1280, "features"),
    "EfficientNetB2": (models.efficientnet_b2, models.EfficientNet_B2_Weights, 1408, "features"),
    "DenseNet121": (models.densenet121, models.DenseNet121_Weights, 1024, "densenet"),
    "DenseNet169": (models.densenet169, models.DenseNet169_Weights, 1664, "densenet"),
    "DenseNet201": (models.densenet201, models.DenseNet201_Weights, 1920, "densenet"),
    "MobileNetV2": (models.mobilenet_v2, models.MobileNet_V2_Weights, 1280, "features"),
    "MobileNetV3Small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights, 576, "features"),
    "MobileNetV3Large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights, 960, "features"),
    "VGG16": (models.vgg16, models.VGG16_Weights, 25088, "vgg"),
    "VGG19": (models.vgg19, models.VGG19_Weights, 25088, "vgg"),
}


class Regressor(nn.Module):
    def __init__(self, name="ResNet18", pretrained=True):
        super().__init__()
        builder, weights, dim, family = _SPECS[name]
        base = builder(weights=weights.DEFAULT if pretrained else None)
        self.family = family

        if family == "resnet":
            base.conv1 = _grayscale_conv(base.conv1)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
            self.pool = nn.Identity()
        elif family == "densenet":
            base.features.conv0 = _grayscale_conv(base.features.conv0)
            self.feature_extractor = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif family == "vgg":
            base.features[0] = _grayscale_conv(base.features[0])
            self.feature_extractor = base.features
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
        else:
            base.features[0][0] = _grayscale_conv(base.features[0][0])
            self.feature_extractor = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if family == "vgg":
            self.fc = _head(dim, hidden=(4096, 1024), dropout=(0.5, 0.5))
        else:
            self.fc = _head(dim)

    def features(self, x):
        z = self.feature_extractor(x)
        if self.family == "densenet":
            z = F.relu(z)
        return torch.flatten(self.pool(z), 1)

    def forward(self, x):
        return self.fc(self.features(x)).squeeze(1)


class SequenceRegressor(nn.Module):
    def __init__(self, encoder="ResNet18", pretrained=True, hidden=256, layers=1):
        super().__init__()
        base = Regressor(encoder, pretrained=pretrained)
        self.feature_extractor = base.feature_extractor
        self.pool = base.pool
        self.family = base.family
        with torch.no_grad():
            dim = self._encode(torch.zeros(1, 1, *IMG_SIZE)).shape[1]
        self.lstm = nn.LSTM(dim, hidden, layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(),
                                nn.Dropout(0.2), nn.Linear(128, 1))

    def _encode(self, x):
        z = self.feature_extractor(x)
        if self.family == "densenet":
            z = F.relu(z)
        return torch.flatten(self.pool(z), 1)

    def forward(self, x):
        b, t = x.shape[:2]
        feats = self._encode(x.flatten(0, 1)).view(b, t, -1)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1]).squeeze(1)


class BranchRegressor(nn.Module):
    def __init__(self, encoder="ResNet18", pretrained=True):
        super().__init__()
        base = Regressor(encoder, pretrained=pretrained)
        self.feature_extractor = base.feature_extractor
        self.pool = base.pool
        self.family = base.family
        head = list(base.fc.children())
        self.shared = nn.Sequential(*head[:-1])
        dim = head[-1].in_features
        self.soc_head = nn.Linear(dim, 1)
        self.branch_head = nn.Linear(dim, 2)

    def forward(self, x):
        z = self.feature_extractor(x)
        if self.family == "densenet":
            z = F.relu(z)
        z = self.shared(torch.flatten(self.pool(z), 1))
        return self.soc_head(z).squeeze(1), self.branch_head(z)


ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish, nn.GELU, nn.LeakyReLU)


def target_layer(model):
    trunk = model.feature_extractor
    if isinstance(trunk, nn.Sequential) and len(trunk) > 2 and isinstance(trunk[2], ACTIVATIONS):
        return trunk[2]
    for module in trunk.modules():
        if isinstance(module, ACTIVATIONS):
            return module
    raise ValueError("no activation layer found")


def disable_inplace(model):
    for module in model.modules():
        if getattr(module, "inplace", False):
            module.inplace = False


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())),
            layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())),
        ]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def _map(self, size, scalar, retain=False):
        self.model.zero_grad(set_to_none=True)
        scalar.backward(retain_graph=retain)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        cam = cam.squeeze().float().cpu().numpy()
        span = float(cam.max() - cam.min())
        return (cam - cam.min()) / (span + 1e-8), span <= 1e-8

    def generate(self, x):
        out = self.model(x)
        return self._map(x.shape[-2:], out.sum())

    def generate_multitask(self, x):
        soc, _ = self.model(x)
        cam_soc, empty_soc = self._map(x.shape[-2:], soc.sum(), retain=True)
        soc2, branch = self.model(x)
        cls = int(branch.argmax(1).item())
        confidence = float(torch.softmax(branch.float(), 1).max().item())
        cam_branch, empty_branch = self._map(x.shape[-2:], branch[0, cls])
        return cam_soc, cam_branch, empty_soc, empty_branch, float(soc.item()), cls, confidence


def metrics(truths, preds):
    truths = np.asarray(truths, dtype=float)
    preds = np.asarray(preds, dtype=float)
    variance = np.var(truths)
    return {
        "n": len(truths),
        "MAE": float(mean_absolute_error(truths, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(truths, preds))),
        "R2": float(r2_score(truths, preds)) if variance > 1e-12 else np.nan,
        "Bias": float(np.mean(preds - truths)),
    }


@torch.inference_mode()
def evaluate(model, loader, use_amp=True, sequence=False):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        x = batch[0].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=use_amp):
            out = model(x)
        preds.append(out.float().cpu().numpy())
        truths.append(batch[1].numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    if not np.isfinite(preds).all():
        raise FloatingPointError("non-finite predictions")
    result = metrics(truths, preds)
    result["preds"] = preds
    result["truths"] = truths
    return result


def train_fold(train_loader, test_loader, name="ResNet18", epochs=NUM_EPOCHS,
               lr=LEARNING_RATE, use_amp=True, pretrained=True, model=None,
               sequence=False, callback=None):
    if model is None:
        model = (SequenceRegressor(name, pretrained=pretrained) if sequence
                 else Regressor(name, pretrained=pretrained))
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = torch.amp.GradScaler(DEVICE.type,
                                  enabled=use_amp and AMP_DTYPE == torch.float16)

    history, best_r2, best_state = [], -float("inf"), None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            x = batch[0].to(DEVICE, non_blocking=True)
            y = batch[1].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        avg = total / len(train_loader)
        scheduler.step(avg)

        result = evaluate(model, test_loader, use_amp)
        history.append({"epoch": epoch + 1, "loss": avg, "MAE": result["MAE"],
                        "RMSE": result["RMSE"], "R2": result["R2"]})
        if result["R2"] > best_r2:
            best_r2 = result["R2"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if callback is not None:
            callback(epoch + 1, history[-1])

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), evaluate(model, test_loader, use_amp)


def loaders(dataset, train_idx, test_idx, batch_size=BATCH_SIZE):
    pin = DEVICE.type == "cuda"
    return (DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                       shuffle=True, pin_memory=pin),
            DataLoader(Subset(dataset, test_idx), batch_size=EVAL_BATCH_SIZE,
                       shuffle=False, pin_memory=pin))


def summarise(df, by=None, columns=("MAE", "RMSE", "R2")):
    if by is None:
        return {f"{c}_{s}": getattr(df[c], s)() for c in columns for s in ("mean", "std")}
    out = df.groupby(by, observed=True)[list(columns)].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def save(df, folder, name):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    df.to_csv(path, index=False)
    return path


def result_dir(*parts):
    path = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path
