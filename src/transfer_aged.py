import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ResNet18")
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=core.LEARNING_RATE)
    parser.add_argument("--relative-soc", action="store_true")
    parser.add_argument("--save-base", action="store_true")
    parser.add_argument("--out", default="transfer")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    fresh_images, fresh_meta = core.load_dataset("fresh")
    aged_images, aged_meta = core.load_dataset("aged")

    fresh_targets = fresh_meta["target"].values.copy()
    aged_targets = aged_meta["target"].values.copy()
    if args.relative_soc:
        fresh_targets = fresh_targets / fresh_targets.max() * 100.0
        aged_targets = aged_targets / aged_targets.max() * 100.0

    fresh_ds = core.FrameDataset(fresh_images, fresh_targets, fresh_meta["filename"].tolist())
    aged_ds = core.FrameDataset(aged_images, aged_targets, aged_meta["filename"].tolist())
    aged_loader = DataLoader(aged_ds, batch_size=core.EVAL_BATCH_SIZE, shuffle=False,
                             pin_memory=core.DEVICE.type == "cuda")
    splits = core.make_splits(fresh_meta, "Leave-one-cycle-out")

    print(f"fresh {len(fresh_ds)} frames, target max {fresh_targets.max():.2f} | "
          f"aged {len(aged_ds)} frames, target max {aged_targets.max():.2f}", flush=True)

    rows = []
    for i, (label, train_idx, test_idx) in enumerate(splits, 1):
        train_loader, test_loader = core.loaders(fresh_ds, train_idx, test_idx,
                                                 args.batch_size)
        t0 = time.perf_counter()
        model, _, fresh_result = core.train_fold(train_loader, test_loader, args.model,
                                                 epochs=args.epochs, lr=args.lr)
        aged_result = core.evaluate(model, aged_loader)
        rows.append({"fold": label, "fresh_MAE": fresh_result["MAE"],
                     "fresh_R2": fresh_result["R2"], "aged_MAE": aged_result["MAE"],
                     "aged_RMSE": aged_result["RMSE"], "aged_R2": aged_result["R2"],
                     "aged_bias": aged_result["Bias"],
                     "aged_pred_std": float(np.std(aged_result["preds"])),
                     "aged_true_std": float(np.std(aged_result["truths"]))})
        print(f"[{i}/{len(splits)}] {label}: fresh MAE {fresh_result['MAE']:.3f} | "
              f"aged MAE {aged_result['MAE']:.3f} R2 {aged_result['R2']:.3f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        core.save(pd.DataFrame(rows), out, "per_fold.csv")
        del model, train_loader, test_loader
        if core.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if args.save_base:
        loader = DataLoader(fresh_ds, batch_size=args.batch_size, shuffle=True,
                            pin_memory=core.DEVICE.type == "cuda")
        model, _, _ = core.train_fold(loader, aged_loader, args.model,
                                      epochs=args.epochs, lr=args.lr)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   os.path.join(out, "base_fresh.pth"))
        print(f"base model trained on all fresh data saved to {out}", flush=True)

    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "MAE_mean": df.aged_MAE.mean(), "MAE_std": df.aged_MAE.std(),
        "RMSE_mean": df.aged_RMSE.mean(), "RMSE_std": df.aged_RMSE.std(),
        "R2_mean": df.aged_R2.mean(), "R2_std": df.aged_R2.std(),
        "bias_mean": df.aged_bias.mean(), "bias_std": df.aged_bias.std(),
        "pred_std": df.aged_pred_std.mean(), "true_std": df.aged_true_std.mean()}])
    core.save(summary, out, "summary.csv")
    print("\n" + summary.round(3).to_string(index=False), flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
