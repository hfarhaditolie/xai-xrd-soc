import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import core

NAMES = {1: "Charging", 0: "Discharging"}


def group_table(df, by, labels=None):
    rows = []
    for key, group in df.groupby(by, observed=True):
        correct = group["correct"].values
        error = group["abs_err"].values
        rows.append({by: key, "n": len(group),
                     "MAE": error.mean(),
                     "RMSE": float(np.sqrt(np.mean((group.pred - group.target) ** 2))),
                     "branch_acc": correct.mean(),
                     "mean_conf": group["confidence"].mean(),
                     "MAE_branch_correct": error[correct].mean() if correct.any() else np.nan,
                     "MAE_branch_wrong": error[~correct].mean() if (~correct).any() else np.nan,
                     "rho_conf_abserr": stats.spearmanr(group["confidence"], error).statistic})
    table = pd.DataFrame(rows)
    if labels is not None:
        table[by] = pd.Categorical(table[by], categories=labels, ordered=True)
        table = table.sort_values(by)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",
                        default=os.path.join(core.RESULTS_DIR, "multitask",
                                             "oof_predictions.csv"))
    parser.add_argument("--out", default="classification")
    args = parser.parse_args()

    out = core.result_dir(args.out)
    df = pd.read_csv(args.predictions)
    df["correct"] = df["branch_pred"] == df["branch_true"]
    df["abs_err"] = (df["pred"] - df["target"]).abs()

    per_branch = []
    for value in (1, 0):
        group = df[df.branch_true == value]
        predicted = df[df.branch_pred == value]
        correct = group["correct"].values
        error = group["abs_err"].values
        per_branch.append({
            "state": NAMES[value], "N": len(group),
            "recall": correct.mean(),
            "precision": (predicted["branch_true"] == value).mean(),
            "conf_correct": group["confidence"].values[correct].mean(),
            "conf_wrong": group["confidence"].values[~correct].mean(),
            "MAE": error.mean(),
            "MAE_branch_correct": error[correct].mean(),
            "MAE_branch_wrong": error[~correct].mean()})

    correct = df["correct"].values
    error = df["abs_err"].values
    per_branch.append({
        "state": "All", "N": len(df), "recall": correct.mean(), "precision": np.nan,
        "conf_correct": df["confidence"].values[correct].mean(),
        "conf_wrong": df["confidence"].values[~correct].mean(),
        "MAE": error.mean(), "MAE_branch_correct": error[correct].mean(),
        "MAE_branch_wrong": error[~correct].mean()})

    branch_table = pd.DataFrame(per_branch)
    core.save(branch_table, out, "per_branch.csv")
    print("per branch\n" + branch_table.round(4).to_string(index=False), flush=True)

    confusion = pd.crosstab(df.branch_true.map(NAMES), df.branch_pred.map(NAMES))
    confusion.to_csv(os.path.join(out, "confusion_matrix.csv"))
    print("\nconfusion matrix\n" + confusion.to_string(), flush=True)

    labels = sorted(df["soc_interval"].dropna().unique(),
                    key=lambda s: float(str(s).split("-")[0]))
    for by, lab in (("rate", None), ("cycle", None), ("soc_interval", labels)):
        table = group_table(df, by, lab)
        core.save(table, out, f"by_{by}.csv")
        print(f"\nby {by}\n" + table.round(4).to_string(index=False), flush=True)

    recall = df.groupby(["cycle", "branch_true"])["correct"].mean().unstack().rename(
        columns=NAMES)
    recall.to_csv(os.path.join(out, "recall_by_cycle.csv"))
    print("\nrecall by cycle\n" + recall.round(3).to_string(), flush=True)
    print(f"\nmean over cycles: charging {recall['Charging'].mean():.4f} +/- "
          f"{recall['Charging'].std():.4f} | discharging "
          f"{recall['Discharging'].mean():.4f} +/- {recall['Discharging'].std():.4f}",
          flush=True)

    interval_recall = df.groupby(["soc_interval", "branch_true"])["correct"].mean().unstack()
    interval_recall = interval_recall.rename(columns=NAMES).reindex(labels)
    interval_recall.to_csv(os.path.join(out, "recall_by_soc_interval.csv"))
    print("\nrecall by SoC interval\n" + interval_recall.round(3).to_string(), flush=True)

    selective = []
    for keep in (1.0, 0.9, 0.8, 0.7):
        threshold = df["confidence"].quantile(1 - keep)
        subset = df[df["confidence"] >= threshold]
        selective.append({"keep": keep, "confidence_threshold": threshold,
                          "MAE": subset["abs_err"].mean(),
                          "branch_acc": subset["correct"].mean()})
    core.save(pd.DataFrame(selective), out, "selective_prediction.csv")
    print("\nselective prediction\n" +
          pd.DataFrame(selective).round(4).to_string(index=False), flush=True)
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
