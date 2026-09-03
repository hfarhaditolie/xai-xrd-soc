import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from skimage.feature import hog
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

ALPHAS = np.logspace(-6, 6, 40)
COMPONENTS = [2, 5, 10, 20, 50]
HOG_KWARGS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                  block_norm="L2-Hys", feature_vector=True)


def peak_features(x, y, n_peaks=3):
    y = y / (np.trapezoid(y, x) + 1e-12)
    idx, props = find_peaks(y, prominence=np.ptp(y) * 0.01, distance=3)
    if len(idx) == 0:
        return np.full(2 * n_peaks, np.nan)
    top = np.sort(idx[np.argsort(props["prominences"])[::-1][:n_peaks]])
    step = x[1] - x[0]
    feats = []
    for j in top:
        shift = 0.0
        if 0 < j < len(y) - 1:
            denom = y[j - 1] - 2 * y[j] + y[j + 1]
            if abs(denom) > 1e-30:
                shift = float(np.clip(0.5 * (y[j - 1] - y[j + 1]) / denom, -1, 1))
        feats += [x[j] + shift * step, y[j]]
    while len(feats) < 2 * n_peaks:
        feats += [np.nan, np.nan]
    return np.asarray(feats[:2 * n_peaks])


def build_peak_matrix(meta):
    rows = []
    for name in meta["filename"]:
        stem = os.path.splitext(name)[0]
        path = os.path.join(core.FRESH["patterns"], stem + "_azimAvg.dat")
        if not os.path.exists(path):
            rows.append(np.full(6, np.nan))
            continue
        x, y = core.read_pattern(path)
        rows.append(peak_features(x, y))
    matrix = np.vstack(rows)
    for c in range(matrix.shape[1]):
        col = matrix[:, c]
        col[np.isnan(col)] = np.nanmedian(col)
    return matrix


def fit_predict(method, n_comp, xtr, ytr, xte):
    if method == "PCA+ridge":
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0).fit(xtr)
        model = RidgeCV(alphas=ALPHAS).fit(pca.transform(xtr), ytr)
        return model.predict(pca.transform(xte))
    if method == "PLS":
        return PLSRegression(n_components=n_comp, scale=False).fit(xtr, ytr).predict(xte).ravel()
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="Leave-one-cycle-out", choices=core.SPLIT_SCHEMES)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--out", default="classical")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    images, meta = core.load_dataset("fresh")
    y = meta["target"].values.astype(np.float64)
    groups = meta["cycle"].values
    splits = core.make_splits(meta, args.scheme)

    pixels = np.stack([im.reshape(-1) for im in images]).astype(np.float32) / 255.0
    t0 = time.perf_counter()
    peaks = build_peak_matrix(meta)
    print(f"peak features {peaks.shape} in {time.perf_counter()-t0:.0f}s", flush=True)
    t0 = time.perf_counter()
    hogs = np.stack([hog(im.astype(np.float32) / 255.0, **HOG_KWARGS) for im in images])
    print(f"HOG features {hogs.shape} in {time.perf_counter()-t0:.0f}s", flush=True)

    rows, curve = [], []
    for label, tr, te in splits:
        xtr, xte, ytr, yte = pixels[tr], pixels[te], y[tr], y[te]

        for method in ("PCA+ridge", "PLS"):
            for nc in COMPONENTS:
                curve.append({"fold": label, "method": method, "n_components": nc,
                              **core.metrics(yte, fit_predict(method, nc, xtr, ytr, xte))})

            inner = GroupKFold(n_splits=min(args.inner_folds, len(np.unique(groups[tr]))))
            scores = {nc: [] for nc in COMPONENTS}
            for itr, iva in inner.split(xtr, ytr, groups=groups[tr]):
                for nc in COMPONENTS:
                    p = fit_predict(method, nc, xtr[itr], ytr[itr], xtr[iva])
                    scores[nc].append(np.mean(np.abs(p - ytr[iva])))
            best = min(COMPONENTS, key=lambda nc: np.mean(scores[nc]))
            rows.append({"method": method, "fold": label, "n_features": xtr.shape[1],
                         "selected_components": best,
                         **core.metrics(yte, fit_predict(method, best, xtr, ytr, xte))})
            print(f"  {label:22s} {method:10s} k={best:3d} MAE {rows[-1]['MAE']:7.3f}",
                  flush=True)

        for feats, name in ((pixels, "Ridge (raw pixels)"), (peaks, "Ridge (1D peak features)"),
                            (hogs, "Ridge (HOG features)")):
            mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-12
            model = RidgeCV(alphas=ALPHAS).fit((feats[tr] - mu) / sd, y[tr])
            pred = model.predict((feats[te] - mu) / sd)
            rows.append({"method": name, "fold": label, "n_features": feats.shape[1],
                         "selected_components": np.nan, **core.metrics(y[te], pred)})
            print(f"  {label:22s} {name:26s} MAE {rows[-1]['MAE']:7.3f}", flush=True)

        core.save(pd.DataFrame(rows), out, "per_fold.csv")
        core.save(pd.DataFrame(curve), out, "component_curve.csv")

    df = pd.DataFrame(rows)
    summary = df.groupby("method").agg(
        n_features=("n_features", "first"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std")).reset_index().sort_values("MAE_mean")
    core.save(summary, out, "summary.csv")
    print("\n" + summary.round(4).to_string(index=False), flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
