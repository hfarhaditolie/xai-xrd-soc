import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import core


@torch.inference_mode()
def mean_squared_error(model, loader, use_amp=True):
    model.eval()
    total, count = 0.0, 0
    for x, y, _ in loader:
        x = x.to(core.DEVICE, non_blocking=True)
        with torch.autocast(core.DEVICE.type, dtype=core.AMP_DTYPE, enabled=use_amp):
            pred = model(x)
        total += float(((pred.float().cpu() - y) ** 2).sum())
        count += len(y)
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ResNet18")
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=core.LEARNING_RATE)
    parser.add_argument("--out", default="loss_curves")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    images, meta = core.load_dataset("fresh")
    dataset = core.FrameDataset(images, meta["target"].values, meta["filename"].tolist())
    splits = core.make_splits(meta, "Leave-one-cycle-out")
    pin = core.DEVICE.type == "cuda"

    rows = []
    for i, (label, train_idx, test_idx) in enumerate(splits, 1):
        train_loader, test_loader = core.loaders(dataset, train_idx, test_idx, args.batch_size)
        train_eval = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx), batch_size=core.EVAL_BATCH_SIZE,
            shuffle=False, pin_memory=pin)

        model = core.Regressor(args.model).to(core.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=core.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                               factor=0.5)
        scaler = torch.amp.GradScaler(core.DEVICE.type,
                                      enabled=core.AMP_DTYPE == torch.float16)
        criterion = nn.MSELoss()
        print(f"[{i}/{len(splits)}] {label}: train {len(train_idx)}, test {len(test_idx)}",
              flush=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for x, y, _ in train_loader:
                x = x.to(core.DEVICE, non_blocking=True)
                y = y.to(core.DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(core.DEVICE.type, dtype=core.AMP_DTYPE):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item()
            running /= len(train_loader)
            scheduler.step(running)
            rows.append({"fold": label, "epoch": epoch, "train_running": running,
                         "train": mean_squared_error(model, train_eval),
                         "test": mean_squared_error(model, test_loader),
                         "lr": optimizer.param_groups[0]["lr"]})
            if epoch % 10 == 0:
                print(f"      epoch {epoch:3d}  train {rows[-1]['train']:8.3f}  "
                      f"test {rows[-1]['test']:8.3f}", flush=True)

        core.save(pd.DataFrame(rows), out, "loss_curves.csv")
        del model, train_loader, test_loader, train_eval
        if core.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    final = df[df.epoch == args.epochs]
    best = df.loc[df.groupby("fold")["test"].idxmin()]
    print(f"\nepoch {args.epochs}: train {final.train.mean():.3f} +/- {final.train.std():.3f} | "
          f"test {final.test.mean():.3f} +/- {final.test.std():.3f}", flush=True)
    print(f"best test loss at epoch {best.epoch.median():.0f} "
          f"(range {best.epoch.min()}-{best.epoch.max()}), value {best.test.mean():.3f}",
          flush=True)
    core.save(best, out, "best_epoch_per_fold.csv")
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
