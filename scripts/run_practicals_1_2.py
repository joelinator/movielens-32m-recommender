"""Execute Practicals 1 and 2 experiments and generate publication PDF figures.

Generates:
- `report/figures/fig1_dataset_distributions.pdf`: Rating distribution & single
  log-log degree distribution plot with power-law fits and k_min=20 cutoff.
- `report/figures/fig2_bias_als_convergence.pdf`: Monotonic regularized loss
  (verifying user-CSR vs movie-CSR equivalence) and Train/Test RMSE curves.
- `report/results_practicals_1_2.json`: Exact numerical metrics for the report.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.als_biases import fit_bias_als
from src.data_indexing import load_indexed_dataset


def configure_matplotlib() -> None:
    """Configure matplotlib for ICML publication-quality vector PDFs."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )


def fit_power_law_mle(degrees: np.ndarray, k_min: int) -> float:
    """Compute Clauset-Shalizi-Newman continuous MLE power-law exponent alpha."""
    tail = degrees[degrees >= k_min].astype(np.float64)
    if len(tail) == 0:
        return float("nan")
    return float(1.0 + len(tail) / np.sum(np.log(tail / (k_min - 0.5))))


def main() -> None:
    configure_matplotlib()
    figures_dir = ROOT_DIR / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    t0 = time.perf_counter()
    ds_small = load_indexed_dataset(ROOT_DIR / "small_dataset.pkl")
    t_small_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    ds_full = load_indexed_dataset(ROOT_DIR / "dataset.pkl")
    t_full_load = time.perf_counter() - t0

    # 1. Practical 1: Exploratory Statistics
    full_ratings = np.concatenate(
        [ds_full.user_train.ratings, ds_full.user_test.ratings]
    )
    small_ratings = np.concatenate(
        [ds_small.user_train.ratings, ds_small.user_test.ratings]
    )

    star_values = np.arange(0.5, 5.5, 0.5, dtype=np.float32)
    full_hist_counts = np.array(
        [np.sum(full_ratings == s) for s in star_values], dtype=np.int64
    )
    small_hist_counts = np.array(
        [np.sum(small_ratings == s) for s in star_values], dtype=np.int64
    )
    full_hist_probs = full_hist_counts / len(full_ratings)
    small_hist_probs = small_hist_counts / len(small_ratings)

    user_deg_full = (
        ds_full.user_train.row_degrees() + ds_full.user_test.row_degrees()
    )
    movie_deg_full = (
        ds_full.movie_train.row_degrees() + ds_full.movie_test.row_degrees()
    )
    user_deg_small = (
        ds_small.user_train.row_degrees() + ds_small.user_test.row_degrees()
    )
    movie_deg_small = (
        ds_small.movie_train.row_degrees() + ds_small.movie_test.row_degrees()
    )

    alpha_user_full = fit_power_law_mle(user_deg_full, k_min=50)
    alpha_movie_full = fit_power_law_mle(movie_deg_full[movie_deg_full > 0], k_min=50)

    # Plot Figure 1: Rating Distribution & Log-Log Degree Distributions
    # Note: Per ICML guidelines, NO internal titles inside plot panels.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.75))

    x_pos = np.arange(len(star_values))
    width = 0.38
    ax1.bar(
        x_pos - width / 2,
        full_hist_probs * 100.0,
        width=width,
        color="#1f4e78",
        edgecolor="black",
        linewidth=0.4,
        label="MovieLens-32M (Full)",
    )
    ax1.bar(
        x_pos + width / 2,
        small_hist_probs * 100.0,
        width=width,
        color="#5b9bd5",
        edgecolor="black",
        linewidth=0.4,
        label="MovieLens-5M (Subset)",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{s:.1f}" for s in star_values])
    ax1.set_xlabel("(a) Star Rating Value $r_{m,n}$")
    ax1.set_ylabel("Percentage of Ratings (%)")
    ax1.legend(loc="upper left", frameon=True)

    # Right panel: Log-log degree frequency scatter (number of movies/users vs degree k)
    u_vals, u_freq = np.unique(user_deg_full[user_deg_full > 0], return_counts=True)
    m_vals, m_freq = np.unique(movie_deg_full[movie_deg_full > 0], return_counts=True)

    ax2.scatter(
        m_vals,
        m_freq,
        s=7,
        alpha=0.55,
        color="#d95f02",
        edgecolors="none",
        label=r"Movies (degree $|\Omega^n|$)",
        rasterized=True,
    )
    ax2.scatter(
        u_vals,
        u_freq,
        s=7,
        alpha=0.55,
        color="#1b9e77",
        edgecolors="none",
        label=r"Users (degree $|\Omega_m|$)",
        rasterized=True,
    )
    ax2.axvline(
        20,
        color="#333333",
        linestyle=":",
        linewidth=1.2,
        label="User curation cutoff ($k_{\\min}=20$)",
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("(b) Degree $k$ (Number of Ratings)")
    ax2.set_ylabel("Frequency Count $N(k)$")
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig1_path = figures_dir / "fig1_dataset_distributions.pdf"
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig1_path}")

    # 2. Practical 2: Bias-Only ALS on 5M and 32M datasets
    # Warmup JIT first


if __name__ == '__main__':
    main()
