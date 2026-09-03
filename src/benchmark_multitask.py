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


def train(loader, epochs, lr, lambda_cls, mean, std, use_amp=True):
    model = core.BranchRegressor().to(core.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=core.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = torch.amp.GradScaler(core.DEVICE.type,
                                  enabled=use_amp and core.AMP_DTYPE == torch.float16)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        total = 0.0
        for x, y, b in loader:
            x = x.to(core.DEVICE, non_blocking=True)
            y = (y.to(core.DEVICE, non_blocking=True) - mean) / std
            b = b.to(core.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(core.DEVICE.type, dtype=core.AMP_DTYPE, enabled=use_amp):
                soc, branch = model(x)
                loss = mse(soc, y) + lambda_cls * ce(branch, b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step(total / len(loader))
    return model


@torch.inference_mode()
def predict(model, loader, mean, std, use_amp=True):
    model.eval()
    socs, branches, confs, ys, bs = [], [], [], [], []
    for x, y, b in loader:
        x = x.to(core.DEVICE, non_blocking=True)
        with torch.autocast(core.DEVICE.type, dtype=core.AMP_DTYPE, enabled=use_amp):
            soc, branch = model(x)
        prob = torch.softmax(branch.float(), 1)
        socs.append((soc.float() * std + mean).cpu().numpy())
        branches.append(prob.argmax(1).cpu().numpy())
        confs.append(prob.max(1).values.cpu().numpy())
        ys.append(y.numpy())
        bs.append(b.numpy())
    return (np.concatenate(socs), np.concatenate(branches), np.concatenate(confs),
            np.concatenate(ys), np.concatenate(bs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="Leave-one-cycle-out", choices=core.SPLIT_SCHEMES)
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=core.LEARNING_RATE)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--out", default="multitask")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    images, meta = core.load_dataset("fresh")
    branch = core.charge_discharge_labels(meta)
    dataset = core.BranchDataset(images, meta["target"].values, branch,
                                 meta["filename"].tolist())
    intervals, _ = core.soc_intervals(meta)
    splits = core.make_splits(meta, args.scheme)
    pin = core.DEVICE.type == "cuda"

    print(f"{len(dataset)} frames | charging {int(branch.sum())} | "
          f"discharging {int((1 - branch).sum())}", flush=True)

    oof_soc = np.full(len(dataset), np.nan)
    oof_conf = np.full(len(dataset), np.nan)
    oof_branch = np.full(len(dataset), -1, dtype=np.int64)
    fold_rows = []

    for i, (label, train_idx, test_idx) in enumerate(splits, 1):
        mean = float(meta["target"].values[train_idx].mean())
        std = float(meta["target"].values[train_idx].std())
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx), batch_size=args.batch_size,
            shuffle=True, pin_memory=pin)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, test_idx), batch_size=core.EVAL_BATCH_SIZE,
            shuffle=False, pin_memory=pin)

        t0 = time.perf_counter()
        model = train(train_loader, args.epochs, args.lr, args.lambda_cls, mean, std)
        soc, pred_branch, conf, y, b = predict(model, test_loader, mean, std)

        oof_soc[test_idx], oof_conf[test_idx] = soc, conf
        oof_branch[test_idx] = pred_branch
        correct = pred_branch == b
        error = np.abs(soc - y)
        fold_rows.append({"fold": label, "cycle": int(meta["cycle"].values[test_idx][0]),
                          "rate": meta["rate"].values[test_idx][0], "n": len(y),
                          **{k: v for k, v in core.metrics(y, soc).items() if k != "n"},
                          "branch_acc": float(correct.mean()),
                          "mean_conf": float(conf.mean()),
                          "MAE_branch_correct": float(error[correct].mean()),
                          "MAE_branch_wrong": float(error[~correct].mean())
                          if (~correct).any() else np.nan})
        print(f"[{i}/{len(splits)}] {label}: MAE {fold_rows[-1]['MAE']:.3f} "
              f"R2 {fold_rows[-1]['R2']:.4f} | branch acc {correct.mean():.3f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        core.save(pd.DataFrame(fold_rows), out, "per_fold.csv")
        del model, train_loader, test_loader
        if core.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    frame = meta.copy()
    frame["pred"] = oof_soc
    frame["confidence"] = oof_conf
    frame["branch_true"] = branch
    frame["branch_pred"] = oof_branch
    frame["soc_interval"] = intervals
    frame["abs_err"] = np.abs(frame["pred"] - frame["target"])
    core.save(frame, out, "oof_predictions.csv")

    folds = pd.DataFrame(fold_rows)
    print(f"\nMAE {folds.MAE.mean():.3f} +/- {folds.MAE.std():.3f} | "
          f"R2 {folds.R2.mean():.4f} +/- {folds.R2.std():.4f} | "
          f"branch accuracy {folds.branch_acc.mean():.4f} +/- {folds.branch_acc.std():.4f}",
          flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
