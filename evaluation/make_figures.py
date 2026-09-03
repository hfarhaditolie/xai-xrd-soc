import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                            "legend.fontsize": 10, "axes.titlesize": 11.5})
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import core

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]


def save(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.{{png,pdf}}", flush=True)


def loss_curve(source, out):
    path = os.path.join(source, "loss_curves", "loss_curves.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    grouped = df.groupby("epoch")
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for key, label, colour in (("train", "Training", PALETTE[0]),
                               ("test", "Test", PALETTE[3])):
        median = grouped[key].median()
        ax.plot(median.index, median, color=colour, lw=2, label=label)
        ax.fill_between(median.index, grouped[key].quantile(0.25),
                        grouped[key].quantile(0.75), color=colour, alpha=0.2, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training and test loss (median and interquartile range across folds)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    save(fig, out, "loss_curve")


def attention_spectrum(source, out):
    folder = os.path.join(source, "gradcam")
    frames = os.path.join(folder, "per_frame.csv")
    if not os.path.exists(frames):
        return
    df = pd.read_csv(frames)
    spectra = np.load(os.path.join(folder, "spectra.npy"))
    angles = np.load(os.path.join(folder, "row_2theta.npy"))
    peaks = sorted(float(re.match(r"P\d_([\d.]+)deg", c).group(1))
                   for c in df.columns if re.match(r"P\d_", c))
    labels = sorted(df["soc_interval"].dropna().unique(),
                    key=lambda s: float(str(s).split("-")[0]))
    domains = list(df["domain"].unique())

    fig, axes = plt.subplots(1, len(domains), figsize=(7 * len(domains), 5), sharey=True,
                             squeeze=False)
    for ax, domain in zip(axes[0], domains):
        for label, colour in zip(labels, PALETTE):
            mask = ((df.domain == domain) & (df.soc_interval == label)).values
            if mask.sum() == 0:
                continue
            ax.plot(angles, spectra[mask].mean(axis=0), lw=1.7, color=colour,
                    label=f"SoC {label}  (n={mask.sum()})")
        for peak in peaks:
            ax.axvspan(peak - 0.3, peak + 0.3, color="tab:orange", alpha=0.1, zorder=0)
        enrichment = df.loc[df.domain == domain, "enrichment"].mean()
        ax.set_xlabel(r"2$\theta$ (deg)")
        ax.set_title(f"{domain} cell   (enrichment {enrichment:.2f}$\\times$)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Attention density (normalised)")
    plt.suptitle("Grad-CAM attention projected onto the diffraction angle",
                 fontweight="bold")
    plt.tight_layout()
    save(fig, out, "attention_spectrum")


def finetune_curve(source, out):
    folder = os.path.join(source, "finetune_fractions")
    path = os.path.join(folder, "summary.csv")
    if not os.path.exists(path):
        return
    summary = pd.read_csv(path)
    zero = pd.read_csv(os.path.join(folder, "zeroshot_per_cycle.csv"))
    names = {"none": "All layers", "stem+layer1": "Stem+L1 frozen",
             "stem+layer1+layer2": "Stem+L1+L2 frozen",
             "head only": "Encoder frozen (head only)"}
    styles = {"none": dict(color="#0072B2", marker="o", ls="-"),
              "stem+layer1": dict(color="#009E73", marker="s", ls="--"),
              "stem+layer1+layer2": dict(color="#E69F00", marker="^", ls="-."),
              "head only": dict(color="#D55E00", marker="D", ls="-")}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for key in summary["freeze"].unique():
        sub = summary[summary.freeze == key].sort_values("fraction")
        axes[0].errorbar(sub.fraction * 100, sub.MAE_mean, yerr=sub.MAE_std, capsize=3,
                         lw=1.8, markersize=5.5, label=names.get(key, key),
                         **styles.get(key, {}))
        axes[1].errorbar(sub.fraction * 100, sub.R2_mean, yerr=sub.R2_std, capsize=3,
                         lw=1.8, markersize=5.5, label=names.get(key, key),
                         **styles.get(key, {}))
    axes[0].axhline(zero.MAE.mean(), color="0.35", ls=":", lw=1.5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("(a) Prediction error")
    axes[1].axhline(zero.R2.mean(), color="0.35", ls=":", lw=1.5)
    axes[1].axhline(0, color="k", lw=0.8, alpha=0.5)
    axes[1].set_ylabel("$R^2$")
    axes[1].set_title("(b) Coefficient of determination")
    for ax in axes:
        ax.set_xlabel("Fine-tuning data (% of aged dataset)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    plt.tight_layout()
    save(fig, out, "finetune_fractions")


def backbone_comparison(source, out):
    path = os.path.join(source, "backbones", "summary.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path).sort_values("MAE_mean")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(len(df))
    for ax, (col, err, title) in zip(axes, [("MAE_mean", "MAE_std", "MAE"),
                                            ("RMSE_mean", "RMSE_std", "RMSE"),
                                            ("R2_mean", "R2_std", "$R^2$")]):
        ax.bar(x, df[col], yerr=df[err], capsize=5, color=PALETTE[0], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=30, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} (mean $\\pm$ s.d. across folds)")
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, out, "backbone_comparison")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=core.RESULTS_DIR)
    parser.add_argument("--out", default=os.path.join(core.RESULTS_DIR, "figures"))
    args = parser.parse_args()
    print(f"writing figures to {args.out}", flush=True)
    for fn in (loss_curve, attention_spectrum, finetune_curve, backbone_comparison):
        fn(args.source, args.out)


if __name__ == "__main__":
    main()
