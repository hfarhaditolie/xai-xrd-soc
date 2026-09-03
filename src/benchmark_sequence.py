import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            conv_block(32, 64), conv_block(64, 128), conv_block(128, 256),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.feature_extractor(x)


class SimpleSequenceRegressor(nn.Module):
    def __init__(self, hidden=256, layers=1):
        super().__init__()
        self.feature_extractor = SimpleEncoder().feature_extractor
        with torch.no_grad():
            dim = torch.flatten(self.feature_extractor(torch.zeros(1, 1, *core.IMG_SIZE)), 1).shape[1]
        self.lstm = nn.LSTM(dim, hidden, layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(),
                                nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        b, t = x.shape[:2]
        feats = torch.flatten(self.feature_extractor(x.flatten(0, 1)), 1).view(b, t, -1)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1]).squeeze(1)


def build(variant):
    if variant == "resnet18-imagenet":
        return core.SequenceRegressor("ResNet18", pretrained=True)
    if variant == "resnet18-scratch":
        return core.SequenceRegressor("ResNet18", pretrained=False)
    if variant == "simplecnn":
        return SimpleSequenceRegressor()
    raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+",
                        default=["resnet18-imagenet", "resnet18-scratch", "simplecnn"])
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--scheme", default="Leave-one-cycle-out", choices=core.SPLIT_SCHEMES)
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=core.LEARNING_RATE)
    parser.add_argument("--out", default="sequence")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    images, meta = core.load_dataset("fresh")
    windows = core.build_windows(meta, args.seq_len)
    dataset = core.SequenceDataset(images, meta["target"].values,
                                   meta["filename"].tolist(), windows)
    splits = core.make_splits(meta, args.scheme)
    print(f"{len(dataset)} windows of {args.seq_len} frames | {len(splits)} folds", flush=True)

    fold_rows, summary_rows = [], []
    for variant in args.variants:
        print(f"\n{variant}", flush=True)
        oof = np.full(len(dataset), np.nan)
        params = None
        for i, (label, train_idx, test_idx) in enumerate(splits, 1):
            train_loader, test_loader = core.loaders(dataset, train_idx, test_idx,
                                                     args.batch_size)
            t0 = time.perf_counter()
            model, _, result = core.train_fold(train_loader, test_loader,
                                               epochs=args.epochs, lr=args.lr,
                                               model=build(variant))
            elapsed = time.perf_counter() - t0
            if params is None:
                params = sum(p.numel() for p in model.parameters())
            oof[test_idx] = result["preds"]
            fold_rows.append({"variant": variant, "fold": label,
                              "cycle": int(meta["cycle"].values[test_idx][0]),
                              "rate": meta["rate"].values[test_idx][0],
                              "MAE": result["MAE"], "RMSE": result["RMSE"],
                              "R2": result["R2"], "train_time_s": elapsed})
            print(f"  [{i}/{len(splits)}] {label}: MAE {result['MAE']:.3f} "
                  f"R2 {result['R2']:.4f} ({elapsed:.0f}s)", flush=True)
            core.save(pd.DataFrame(fold_rows), out, "per_fold.csv")
            del model, train_loader, test_loader
            if core.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        sub = pd.DataFrame([r for r in fold_rows if r["variant"] == variant])
        summary_rows.append({"variant": variant, "params_M": params / 1e6,
                             "MAE_mean": sub.MAE.mean(), "MAE_std": sub.MAE.std(),
                             "RMSE_mean": sub.RMSE.mean(), "RMSE_std": sub.RMSE.std(),
                             "R2_mean": sub.R2.mean(), "R2_std": sub.R2.std()})
        np.save(os.path.join(out, f"oof_{variant}.npy"), oof)
        core.save(pd.DataFrame(summary_rows), out, "summary.csv")

    print("\n" + pd.DataFrame(summary_rows).round(4).to_string(index=False), flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
