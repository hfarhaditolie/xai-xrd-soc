import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import core

HALF_WIDTH = 0.30
N_PEAKS = 6


def mean_pattern(meta, stride=20):
    total, angles, count = None, None, 0
    for name in meta["filename"][::stride]:
        stem = os.path.splitext(name)[0]
        path = os.path.join(core.FRESH["patterns"], stem + "_azimAvg.dat")
        if not os.path.exists(path):
            continue
        x, y = core.read_pattern(path)
        y = y / np.trapezoid(y, x)
        total = y if total is None else total + y
        angles, count = x, count + 1
    return angles, total / max(count, 1)


def find_windows(angles, pattern, n_peaks=N_PEAKS):
    idx, props = find_peaks(pattern, prominence=np.ptp(pattern) * 0.03, distance=4)
    top = idx[np.argsort(props["prominences"])[::-1][:n_peaks]]
    return sorted(float(angles[j]) for j in top)


def indicators(cam, row_angles, masks, any_mask, covered):
    profile = cam.sum(axis=1)
    total = profile.sum()
    if total <= 0:
        return None
    spectrum = profile / total
    values = {"attn_peaks_total": float(spectrum[any_mask].sum()),
              "enrichment": float(spectrum[any_mask].sum() / covered),
              "attn_weighted_2theta": float((spectrum * row_angles).sum())}
    return spectrum, values


def attribute(model, dataset, indices, meta, intervals, row_angles, masks, any_mask,
              covered, peaks, domain, rows, spectra):
    model.eval()
    core.disable_inplace(model)
    layer = core.target_layer(model)
    for i in indices:
        x, y, name = dataset[int(i)]
        try:
            with core.GradCAM(model, layer) as cam_hook:
                cam, empty = cam_hook.generate(x.unsqueeze(0).to(core.DEVICE))
        except Exception:
            continue
        if empty:
            continue
        result = indicators(cam, row_angles, masks, any_mask, covered)
        if result is None:
            continue
        spectrum, values = result
        record = {"domain": domain, "filename": name,
                  "cycle": int(meta["cycle"].values[i]),
                  "rate": meta["rate"].values[i],
                  "soc_interval": intervals[i], "target": float(y), **values}
        for k, p in enumerate(peaks, 1):
            record[f"P{k}_{p:.2f}deg"] = float(spectrum[masks[k - 1]].sum())
        rows.append(record)
        spectra.append(spectrum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ResNet18")
    parser.add_argument("--aged-models", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=core.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.BATCH_SIZE)
    parser.add_argument("--out", default="gradcam")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    checkpoints = os.path.join(out, "checkpoints")
    os.makedirs(checkpoints, exist_ok=True)

    fresh_images, fresh_meta = core.load_dataset("fresh")
    fresh_ds = core.FrameDataset(fresh_images, fresh_meta["target"].values,
                                 fresh_meta["filename"].tolist())
    fresh_intervals, labels = core.soc_intervals(fresh_meta)

    angles, pattern = mean_pattern(fresh_meta)
    peaks = find_windows(angles, pattern)
    row_angles = np.linspace(core.TTH_MIN, core.TTH_MAX, core.IMG_SIZE[0])
    masks = [(row_angles >= p - HALF_WIDTH) & (row_angles <= p + HALF_WIDTH) for p in peaks]
    any_mask = np.any(np.stack(masks), axis=0)
    covered = sum(2 * HALF_WIDTH for _ in peaks) / (core.TTH_MAX - core.TTH_MIN)
    print(f"peaks at {[f'{p:.2f}' for p in peaks]} covering {covered:.1%} of the range",
          flush=True)

    rows, spectra = [], []
    for i, (label, train_idx, test_idx) in enumerate(core.make_splits(fresh_meta), 1):
        tag = re.sub(r"\W+", "_", label).strip("_")
        path = os.path.join(checkpoints, f"{tag}.pth")
        if os.path.exists(path):
            model = core.Regressor(args.model).to(core.DEVICE)
            model.load_state_dict(torch.load(path, map_location=core.DEVICE))
        else:
            train_loader, test_loader = core.loaders(fresh_ds, train_idx, test_idx,
                                                     args.batch_size)
            model, _, result = core.train_fold(train_loader, test_loader, args.model,
                                               epochs=args.epochs)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
            print(f"[{i}/8] {label}: trained, MAE {result['MAE']:.3f}", flush=True)
        attribute(model, fresh_ds, test_idx, fresh_meta, fresh_intervals, row_angles,
                  masks, any_mask, covered, peaks, "fresh", rows, spectra)
        del model
        if core.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if args.aged_models:
        aged_images, aged_meta = core.load_dataset("aged")
        aged_ds = core.FrameDataset(aged_images, aged_meta["target"].values,
                                    aged_meta["filename"].tolist())
        aged_intervals, _ = core.soc_intervals(aged_meta)
        for path in args.aged_models:
            cycle = re.search(r"cycle(\d+)", os.path.basename(path))
            model = core.Regressor(args.model).to(core.DEVICE)
            model.load_state_dict(torch.load(path, map_location=core.DEVICE))
            idx = (aged_meta.index[aged_meta["cycle"] == int(cycle.group(1))].to_numpy()
                   if cycle else np.arange(len(aged_ds)))
            attribute(model, aged_ds, idx, aged_meta, aged_intervals, row_angles, masks,
                      any_mask, covered, peaks, "aged", rows, spectra)
            print(f"aged: {len(idx)} frames attributed from {os.path.basename(path)}",
                  flush=True)
            del model
            if core.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    core.save(df, out, "per_frame.csv")
    np.save(os.path.join(out, "spectra.npy"), np.stack(spectra))
    np.save(os.path.join(out, "row_2theta.npy"), row_angles)

    peak_cols = [c for c in df.columns if re.match(r"P\d_", c)]
    report = ["attn_peaks_total", "enrichment", "attn_weighted_2theta"] + peak_cols
    for domain in df["domain"].unique():
        sub = df[df.domain == domain]
        for by, fname in (("soc_interval", "by_soc_interval"), ("rate", "by_c_rate"),
                          ("cycle", "by_cycle")):
            table = sub.groupby(by, observed=True)[report].mean()
            table.insert(0, "n", sub.groupby(by, observed=True).size())
            if by == "soc_interval":
                table = table.reindex([b for b in labels if b in table.index])
            table.to_csv(os.path.join(out, f"{domain}_{fname}.csv"))
            print(f"\n{domain} {by}\n{table.round(4).to_string()}", flush=True)
        t, p = stats.ttest_1samp(sub["enrichment"], 1.0)
        print(f"\n{domain}: enrichment {sub.enrichment.mean():.3f} +/- "
              f"{sub.enrichment.std():.3f} (vs uniform t={t:.1f}, p={p:.2e})", flush=True)

    if df["domain"].nunique() > 1:
        a = df.loc[df.domain == "fresh", "enrichment"]
        b = df.loc[df.domain == "aged", "enrichment"]
        print(f"fresh vs aged: Mann-Whitney p = {stats.mannwhitneyu(a, b).pvalue:.2e}",
              flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
