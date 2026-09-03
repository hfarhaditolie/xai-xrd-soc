import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import core
from benchmark_multitask import train

SIGNAL_PERCENTILE = 90


def quantify(cam, image, radius):
    flat = cam.ravel().astype(np.float64)
    total = flat.sum() + 1e-12
    p = flat / total
    nonzero = p[p > 0]
    signal = image >= np.percentile(image, SIGNAL_PERCENTILE)
    return {"signal_mass": float(cam[signal].sum() / total),
            "entropy": float(-(nonzero * np.log(nonzero)).sum() / np.log(flat.size))
            if nonzero.size else 0.0,
            "radial_com": float((p * radius.ravel()).sum())}


def radius_map(height, width):
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.mgrid[0:height, 0:width]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return r / r.max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=core.LEARNING_RATE)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--out", default="gradcam_multitask")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    checkpoints = os.path.join(out, "checkpoints")
    os.makedirs(checkpoints, exist_ok=True)

    images, meta = core.load_dataset("fresh")
    branch = core.charge_discharge_labels(meta)
    dataset = core.BranchDataset(images, meta["target"].values, branch,
                                 meta["filename"].tolist())
    intervals, labels = core.soc_intervals(meta)
    splits = core.make_splits(meta, "Leave-one-cycle-out")
    radius = radius_map(*core.IMG_SIZE)
    pin = core.DEVICE.type == "cuda"

    rows = []
    for i, (label, train_idx, test_idx) in enumerate(splits, 1):
        tag = re.sub(r"\W+", "_", label).strip("_")
        path = os.path.join(checkpoints, f"{tag}.pth")
        mean = float(meta["target"].values[train_idx].mean())
        std = float(meta["target"].values[train_idx].std())

        model = core.BranchRegressor().to(core.DEVICE)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=core.DEVICE))
        else:
            loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                                shuffle=True, pin_memory=pin)
            model = train(loader, args.epochs, args.lr, args.lambda_cls, mean, std)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        print(f"[{i}/{len(splits)}] {label}", flush=True)

        model.eval()
        core.disable_inplace(model)
        layer = core.target_layer(model)
        for idx in test_idx:
            x, y, b = dataset[int(idx)]
            xb = x.unsqueeze(0).to(core.DEVICE)
            try:
                with core.GradCAM(model, layer) as cam_hook:
                    cam_soc, cam_branch, empty_soc, empty_branch, pred, cls, conf = \
                        cam_hook.generate_multitask(xb)
            except Exception:
                continue
            image = np.clip(x.squeeze().numpy(), 0, 1)
            record = {"fold": label, "cycle": int(meta["cycle"].values[idx]),
                      "rate": meta["rate"].values[idx], "soc_interval": intervals[idx],
                      "branch_true": int(b), "branch_pred": cls, "branch_conf": conf,
                      "target": float(y), "pred": pred * std + mean,
                      "empty_soc": empty_soc, "empty_branch": empty_branch,
                      "cam_correlation": float(np.corrcoef(cam_soc.ravel(),
                                                           cam_branch.ravel())[0, 1])}
            record["abs_err"] = abs(record["pred"] - float(y))
            for key, value in quantify(cam_soc, image, radius).items():
                record[f"soc_{key}"] = value
            for key, value in quantify(cam_branch, image, radius).items():
                record[f"branch_{key}"] = value
            rows.append(record)

        core.save(pd.DataFrame(rows), out, "per_frame.csv")
        del model
        if core.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    live = df[~df.empty_soc & ~df.empty_branch]
    columns = ["soc_signal_mass", "soc_entropy", "soc_radial_com", "branch_signal_mass",
               "branch_entropy", "branch_radial_com", "cam_correlation", "abs_err"]
    live = live.copy()
    live["branch"] = np.where(live.branch_true == 1, "charging", "discharging")

    print(f"\n{len(df)} frames | empty SoC maps {int(df.empty_soc.sum())} | "
          f"empty branch maps {int(df.empty_branch.sum())}", flush=True)
    print(f"mean SoC/branch attribution correlation "
          f"{live.cam_correlation.mean():.3f} +/- {live.cam_correlation.std():.3f}",
          flush=True)

    for by in ("soc_interval", "branch", "rate", "cycle"):
        table = live.groupby(by, observed=True)[columns].mean()
        if by == "soc_interval":
            table = table.reindex([b for b in labels if b in table.index])
        table.to_csv(os.path.join(out, f"by_{by}.csv"))
        print(f"\nby {by}\n{table.round(3).to_string()}", flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
