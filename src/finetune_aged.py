import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

FREEZE_LEVELS = {"none": 0, "stem+layer1": 5, "stem+layer1+layer2": 6, "head only": 9}


def apply_freeze(model, n_frozen):
    frozen = []
    for i, block in enumerate(model.feature_extractor):
        if i < n_frozen:
            for p in block.parameters():
                p.requires_grad = False
            frozen.append(block)
    return frozen


def finetune(model, loader, epochs, lr, frozen=(), use_amp=True):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=core.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(core.DEVICE.type,
                                  enabled=use_amp and core.AMP_DTYPE == torch.float16)
    for _ in range(epochs):
        model.train()
        for block in frozen:
            block.eval()
        for x, y, _ in loader:
            x = x.to(core.DEVICE, non_blocking=True)
            y = y.to(core.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(core.DEVICE.type, dtype=core.AMP_DTYPE, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model


def load_base(path, name):
    model = core.Regressor(name).to(core.DEVICE)
    model.load_state_dict(torch.load(path, map_location=core.DEVICE))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="ResNet18")
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=core.LEARNING_RATE)
    parser.add_argument("--freeze", default="none", choices=list(FREEZE_LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--out", default="finetune")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    images, meta = core.load_dataset("aged")
    dataset = core.FrameDataset(images, meta["target"].values, meta["filename"].tolist())
    intervals, _ = core.soc_intervals(meta)
    cycles = sorted(meta["cycle"].unique())
    n_ft = int(round(args.fraction * len(dataset)))
    base_state = torch.load(args.base, map_location="cpu")
    pin = core.DEVICE.type == "cuda"

    print(f"aged {len(dataset)} frames, {len(cycles)} cycles | "
          f"fine-tuning on {args.fraction:.0%} = {n_ft} frames | freeze: {args.freeze}",
          flush=True)

    rows = []
    for c in cycles:
        test_idx = meta.index[meta["cycle"] == c].to_numpy()
        pool = meta.index[meta["cycle"] != c].to_numpy()
        if n_ft > len(pool):
            continue
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=core.EVAL_BATCH_SIZE,
                                 shuffle=False, pin_memory=pin)

        model = core.Regressor(args.model).to(core.DEVICE)
        model.load_state_dict(base_state)
        zero_shot = core.evaluate(model, test_loader)

        for seed in args.seeds:
            ft_idx = np.random.default_rng(1000 * int(c) + seed).choice(pool, n_ft, replace=False)
            model = core.Regressor(args.model).to(core.DEVICE)
            model.load_state_dict(base_state)
            frozen = apply_freeze(model, FREEZE_LEVELS[args.freeze])
            t0 = time.perf_counter()
            model = finetune(model, DataLoader(Subset(dataset, ft_idx),
                                               batch_size=min(args.batch_size, n_ft),
                                               shuffle=True, pin_memory=pin),
                             args.epochs, args.lr, frozen)
            tuned = core.evaluate(model, test_loader)
            rows.append({"cycle": int(c), "rate": meta["rate"].values[test_idx][0],
                         "seed": seed, "n_finetune": n_ft, "n_test": len(test_idx),
                         "freeze": args.freeze, "fraction": args.fraction,
                         "zeroshot_MAE": zero_shot["MAE"], "zeroshot_R2": zero_shot["R2"],
                         "MAE": tuned["MAE"], "RMSE": tuned["RMSE"], "R2": tuned["R2"],
                         "pred_std": float(np.std(tuned["preds"])),
                         "true_std": float(np.std(tuned["truths"])),
                         "time_s": time.perf_counter() - t0})
            print(f"  cycle {c} seed {seed}: zero-shot MAE {zero_shot['MAE']:.3f} -> "
                  f"fine-tuned {tuned['MAE']:.3f} (R2 {tuned['R2']:.4f})", flush=True)
            if args.save_models and seed == args.seeds[0]:
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                           os.path.join(out, f"finetuned_cycle{int(c)}.pth"))
            del model
            if core.DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        core.save(pd.DataFrame(rows), out, "per_fold.csv")

    df = pd.DataFrame(rows)
    per_cycle = df.groupby("cycle")[["MAE", "RMSE", "R2", "zeroshot_MAE"]].mean()
    summary = pd.DataFrame([{
        "fraction": args.fraction, "freeze": args.freeze,
        "zeroshot_MAE_mean": per_cycle.zeroshot_MAE.mean(),
        "MAE_mean": per_cycle.MAE.mean(), "MAE_std": per_cycle.MAE.std(),
        "RMSE_mean": per_cycle.RMSE.mean(), "RMSE_std": per_cycle.RMSE.std(),
        "R2_mean": per_cycle.R2.mean(), "R2_std": per_cycle.R2.std()}])
    core.save(summary, out, "summary.csv")
    print("\n" + summary.round(4).to_string(index=False), flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
