import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def measure_latency(model, batch_sizes=(1, 32), iters=50, warmup=10):
    model.eval()
    stats = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 1, *core.IMG_SIZE, device=core.DEVICE)
        times = []
        with torch.inference_mode():
            for _ in range(warmup):
                with torch.autocast(core.DEVICE.type, dtype=core.AMP_DTYPE):
                    model(x)
            if core.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            for _ in range(iters):
                t0 = time.perf_counter()
                with torch.autocast(core.DEVICE.type, dtype=core.AMP_DTYPE):
                    model(x)
                if core.DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        median = float(np.median(times))
        stats[f"latency_bs{bs}_ms"] = median * 1000
        stats[f"throughput_bs{bs}_fps"] = bs / median
        del x
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["DenseNet121", "EfficientNetB0", "MobileNetV2",
                                 "VGG16", "ResNet18"])
    parser.add_argument("--scheme", default="Leave-one-cycle-out", choices=core.SPLIT_SCHEMES)
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=core.LEARNING_RATE)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out", default="backbones")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    images, meta = core.load_dataset("fresh")
    dataset = core.FrameDataset(images, meta["target"].values, meta["filename"].tolist())
    splits = core.make_splits(meta, args.scheme)
    print(f"{len(dataset)} frames | {len(splits)} folds | device {core.DEVICE}", flush=True)

    fold_rows, model_rows = [], []
    for name in args.models:
        print(f"\n{name}", flush=True)
        oof = np.full(len(dataset), np.nan)
        latency, params = None, None
        started = time.perf_counter()

        for i, (label, train_idx, test_idx) in enumerate(splits, 1):
            train_loader, test_loader = core.loaders(dataset, train_idx, test_idx,
                                                     args.batch_size)
            t0 = time.perf_counter()
            model, _, result = core.train_fold(train_loader, test_loader, name,
                                               epochs=args.epochs, lr=args.lr,
                                               pretrained=not args.no_pretrained)
            elapsed = time.perf_counter() - t0

            if params is None:
                params = sum(p.numel() for p in model.parameters())
            if latency is None:
                latency = measure_latency(model)

            oof[test_idx] = result["preds"]
            fold_rows.append({"model": name, "fold": label,
                              "cycle": int(meta["cycle"].values[test_idx][0]),
                              "rate": meta["rate"].values[test_idx][0],
                              "n_test": len(test_idx), "MAE": result["MAE"],
                              "RMSE": result["RMSE"], "R2": result["R2"],
                              "train_time_s": elapsed})
            print(f"  [{i}/{len(splits)}] {label}: MAE {result['MAE']:.3f} "
                  f"R2 {result['R2']:.4f} ({elapsed:.0f}s)", flush=True)
            core.save(pd.DataFrame(fold_rows), out, "per_fold.csv")
            del model, train_loader, test_loader
            if core.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        sub = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        pooled = core.metrics(meta["target"].values, oof)
        summary = {"model": name, "params_M": params / 1e6,
                   "MAE_mean": sub.MAE.mean(), "MAE_std": sub.MAE.std(),
                   "RMSE_mean": sub.RMSE.mean(), "RMSE_std": sub.RMSE.std(),
                   "R2_mean": sub.R2.mean(), "R2_std": sub.R2.std(),
                   "pooled_MAE": pooled["MAE"], "pooled_R2": pooled["R2"],
                   "train_time_per_fold_s": sub.train_time_s.mean()}
        summary.update(latency)
        model_rows.append(summary)
        np.save(os.path.join(out, f"oof_{name}.npy"), oof)
        core.save(pd.DataFrame(model_rows), out, "summary.csv")
        print(f"  MAE {summary['MAE_mean']:.3f} +/- {summary['MAE_std']:.3f} | "
              f"R2 {summary['R2_mean']:.4f} | {time.perf_counter()-started:.0f}s", flush=True)

    print("\n" + pd.DataFrame(model_rows).round(4).to_string(index=False), flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
