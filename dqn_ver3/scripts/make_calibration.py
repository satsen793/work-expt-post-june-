import argparse
import os
import csv
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_steps(path: str) -> List[Tuple[float, int]]:
    # Returns list of (pred_mastery, correct) for question steps
    data = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("action_type") == "question":
                try:
                    pred = float(row.get("mean_mastery", 0.0))
                    corr_field = row.get("correct")
                    # May be '' for content rows; for questions it's bool-like
                    if corr_field in ("True", "true", "1", 1, True):
                        correct = 1
                    else:
                        correct = 0
                    data.append((pred, correct))
                except Exception:
                    continue
    return data


def reliability_curve(points: List[Tuple[float, int]], bins: int = 10):
    if not points:
        return np.array([]), np.array([])
    preds = np.array([p[0] for p in points])
    labels = np.array([p[1] for p in points])
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    emp_acc = []
    for i in range(bins):
        mask = (preds >= edges[i]) & (preds < edges[i + 1])
        if mask.sum() == 0:
            emp_acc.append(np.nan)
        else:
            emp_acc.append(labels[mask].mean())
    emp_acc = np.array(emp_acc)
    return bin_centers, emp_acc


def plot_calibration(fig_path: str, centers: np.ndarray, emp_acc: np.ndarray):
    ensure_dir(fig_path)
    plt.figure(figsize=(5.5, 5))
    # Perfect calibration line
    x = np.linspace(0, 1, 101)
    plt.plot(x, x, "--", color="gray", label="Ideal")
    # Empirical
    plt.plot(centers, emp_acc, "o-", color="#e45756", label="DQN")
    plt.xlabel("Predicted Mastery (mean)")
    plt.ylabel("Empirical Correctness")
    plt.title("Calibration of Mastery vs Correctness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Make calibration curve from per-step CSV")
    parser.add_argument("--steps-csv", type=str, required=True)
    parser.add_argument("--fig", type=str, default="figures/calibration_curve.png")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    points = load_steps(args.steps_csv)
    centers, emp_acc = reliability_curve(points, bins=args.bins)
    # Fill NaNs with interpolation if needed
    if centers.size == 0:
        print("No data for calibration.")
        return
    m = ~np.isnan(emp_acc)
    if not m.any():
        print("Calibration bins empty.")
        return
    # Simple fill: nearest non-nan
    for i in range(len(emp_acc)):
        if np.isnan(emp_acc[i]):
            # find nearest available
            left = next((j for j in range(i - 1, -1, -1) if not np.isnan(emp_acc[j])), None)
            right = next((j for j in range(i + 1, len(emp_acc)) if not np.isnan(emp_acc[j])), None)
            if left is not None and right is not None:
                emp_acc[i] = 0.5 * (emp_acc[left] + emp_acc[right])
            elif left is not None:
                emp_acc[i] = emp_acc[left]
            elif right is not None:
                emp_acc[i] = emp_acc[right]
            else:
                emp_acc[i] = 0.0
    plot_calibration(args.fig, centers, emp_acc)
    print("Wrote:", args.fig)


if __name__ == "__main__":
    main()
