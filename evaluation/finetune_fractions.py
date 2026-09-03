import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import core
from finetune_aged import FREEZE_LEVELS, apply_freeze, finetune


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="ResNet18")
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.02, 0.05, 0.10, 0.15, 0.25, 0.50])
    parser.add_argument("--freeze", nargs="+", default=list(FREEZE_LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=core.LEARNING_RATE)
    parser.add_argument("--out", default="finetune_fractions")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    images, meta = core.load_dataset("aged")
    dataset = core.FrameDataset(images, meta["target"].values, meta["filename"].tolist())
    cycles = sorted(meta["cycle"].unique())
    base_state = torch.load(args.base, map_location="cpu")
    pin = core.DEVICE.type == "cuda"

    print(f"aged {len(dataset)} frames | {len(cycles)} cycles | "
          f"{len(args.fractions)} fractions x {len(args.freeze)} freezing x "
          f"{len(args.seeds)} seeds x {len(cycles)} folds", flush=True)

    rows, zero_rows, started = [], [], time.perf_counter()
    for c in cycles:
        test_idx = meta.index[meta["cycle"] == c].to_numpy()
        pool = meta.index[meta["cycle"] != c].to_numpy()
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=core.EVAL_BATCH_SIZE,
                                 shuffle=False, pin_memory=pin)

        model = core.Regressor(args.model).to(core.DEVICE)
        model.load_state_dict(base_state)
        zero = core.evaluate(model, test_loader)
        zero_rows.append({"cycle": int(c), "n_test": len(test_idx),
                          "MAE": zero["MAE"], "RMSE": zero["RMSE"], "R2": zero["R2"]})
        print(f"\ncycle {c}: zero-shot MAE {zero['MAE']:.3f} R2 {zero['R2']:.3f}", flush=True)
        del model
        if core.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        for fraction in args.fractions:
            n_ft = int(round(fraction * len(dataset)))
            if n_ft > len(pool):
                continue
            for freeze in args.freeze:
                for seed in args.seeds:
                    ft_idx = np.random.default_rng(1000 * int(c) + seed).choice(
                        pool, n_ft, replace=False)
                    model = core.Regressor(args.model).to(core.DEVICE)
                    model.load_state_dict(base_state)
                    frozen = apply_freeze(model, FREEZE_LEVELS[freeze])
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    model = finetune(model, DataLoader(Subset(dataset, ft_idx),
                                                       batch_size=min(args.batch_size, n_ft),
                                                       shuffle=True, pin_memory=pin),
                                     args.epochs, args.lr, frozen)
                    result = core.evaluate(model, test_loader)
                    rows.append({"cycle": int(c), "fraction": fraction, "n_finetune": n_ft,
                                 "freeze": freeze, "seed": seed,
                                 "trainable_M": trainable / 1e6, "MAE": result["MAE"],
                                 "RMSE": result["RMSE"], "R2": result["R2"]})
                    del model
                    if core.DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
                sub = pd.DataFrame(rows)
                sub = sub[(sub.cycle == int(c)) & (sub.fraction == fraction) &
                          (sub.freeze == freeze)]
                print(f"  {fraction:5.0%} ({n_ft:3d}) {freeze:20s} MAE {sub.MAE.mean():7.3f} "
                      f"R2 {sub.R2.mean():7.3f}", flush=True)
                core.save(pd.DataFrame(rows), out, "raw.csv")
        print(f"  [elapsed {(time.perf_counter()-started)/60:.1f} min]", flush=True)

    df = pd.DataFrame(rows)
    core.save(pd.DataFrame(zero_rows), out, "zeroshot_per_cycle.csv")
    per_cycle = df.groupby(["fraction", "n_finetune", "freeze", "cycle"]).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), R2=("R2", "mean")).reset_index()
    summary = per_cycle.groupby(["fraction", "n_finetune", "freeze"]).agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std")).reset_index()
    core.save(per_cycle, out, "per_cycle.csv")
    core.save(summary, out, "summary.csv")

    print("\n" + summary.round(3).to_string(index=False), flush=True)
    print(f"\ntotal {(time.perf_counter()-started)/60:.1f} min | saved to {out}", flush=True)


if __name__ == "__main__":
    main()
