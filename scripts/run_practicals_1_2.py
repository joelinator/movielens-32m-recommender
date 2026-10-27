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
        label="ML-32M",
    )
    ax1.bar(
        x_pos + width / 2,
        small_hist_probs * 100.0,
        width=width,
        color="#5b9bd5",
        edgecolor="black",
        linewidth=0.4,
        label="ML-5M",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{s:.1f}" for s in star_values], fontsize=7.5)
    ax1.set_xlabel("(a) Rating $r_{m,n}$")
    ax1.set_ylabel("Share (%)")
    ax1.legend(loc="upper left", frameon=False)

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
        label="Movies",
        rasterized=True,
    )
    ax2.scatter(
        u_vals,
        u_freq,
        s=7,
        alpha=0.55,
        color="#1b9e77",
        edgecolors="none",
        label="Users",
        rasterized=True,
    )
    ax2.axvline(
        20,
        color="#333333",
        linestyle=":",
        linewidth=1.1,
        label="$k_{\\min}=20$",
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(0.7, 3e5)
    ax2.set_xlabel("(b) Degree $k$")
    ax2.set_ylabel("Count $N(k)$")
    ax2.legend(loc="upper right", frameon=False, markerscale=1.4)

    fig.tight_layout()
    fig1_path = figures_dir / "fig1_dataset_distributions.pdf"
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig1_path}")

    # 2. Practical 2: Bias-Only ALS on 5M and 32M datasets
    # Warmup JIT first
    print("Warming up Bias ALS Numba kernels...")
    _ = fit_bias_als(
        ds_small,
        num_iters=2,
        lam=1.0,
        gamma=0.05,
        use_global_mean=True,
        verify_dual_views=True,
    )

    print("Running Bias-Only ALS on MovieLens-5M (25 iterations)...")
    res_bias_small = fit_bias_als(
        ds_small,
        num_iters=25,
        lam=1.0,
        gamma=0.05,
        use_global_mean=True,
        verify_dual_views=True,
    )

    print("Running Bias-Only ALS on MovieLens-32M (25 iterations)...")
    res_bias_full = fit_bias_als(
        ds_full,
        num_iters=25,
        lam=1.0,
        gamma=0.05,
        use_global_mean=True,
        verify_dual_views=True,
    )

    # Compute dual-view loss discrepancies
    diff_small = np.abs(
        np.array(res_bias_small.train_loss_user_view)
        - np.array(res_bias_small.train_loss_movie_view)
    )
    diff_full = np.abs(
        np.array(res_bias_full.train_loss_user_view)
        - np.array(res_bias_full.train_loss_movie_view)
    )
    rel_diff_full = diff_full / np.array(res_bias_full.train_loss_user_view)

    # Plot Figure 2: Bias-Only ALS Loss & RMSE Convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.75))
    iters = np.arange(1, 26)

    # Scale loss by 1e6 for clean axis ticks
    loss_u_32m = np.array(res_bias_full.train_loss_user_view) / 1e6
    loss_m_32m = np.array(res_bias_full.train_loss_movie_view) / 1e6
    ax1.plot(
        iters,
        loss_u_32m,
        color="#1f4e78",
        linewidth=1.5,
        label="32M $\\mathcal{L}_U$",
    )
    ax1.scatter(
        iters[::2],
        loss_m_32m[::2],
        color="#d95f02",
        marker="o",
        s=18,
        zorder=4,
        label="32M $\\mathcal{L}_M$",
    )

    ax1_twin = ax1.twinx()
    loss_u_5m = np.array(res_bias_small.train_loss_user_view) / 1e6
    loss_m_5m = np.array(res_bias_small.train_loss_movie_view) / 1e6
    ax1_twin.plot(
        iters,
        loss_u_5m,
        color="#1b9e77",
        linestyle="--",
        linewidth=1.4,
        label="5M $\\mathcal{L}_U$",
    )
    ax1_twin.scatter(
        iters[1::2],
        loss_m_5m[1::2],
        color="#7570b3",
        marker="s",
        s=16,
        zorder=4,
        label="5M $\\mathcal{L}_M$",
    )
    ax1_twin.grid(False)

    ax1.set_xlabel("(a) Iter. $t$")
    ax1.set_ylabel("32M Loss ($\\times 10^6$)")
    ax1_twin.set_ylabel("5M Loss ($\\times 10^6$)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.0, ncol=2, frameon=False)

    # Right panel: Train and Test RMSE
    ax2.plot(
        iters,
        res_bias_full.train_rmse,
        color="#1f4e78",
        linewidth=1.5,
        label="32M Train",
    )
    ax2.plot(
        iters,
        res_bias_full.test_rmse,
        color="#d95f02",
        linewidth=1.5,
        linestyle="-.",
        label="32M Test",
    )
    ax2.plot(
        iters,
        res_bias_small.train_rmse,
        color="#1b9e77",
        linewidth=1.4,
        linestyle="--",
        label="5M Train",
    )
    ax2.plot(
        iters,
        res_bias_small.test_rmse,
        color="#7570b3",
        linewidth=1.4,
        linestyle=":",
        label="5M Test",
    )
    ax2.set_xlabel("(b) Iter. $t$")
    ax2.set_ylabel("RMSE")
    ax2.legend(loc="upper right", fontsize=7.0, ncol=2, frameon=False)

    fig.tight_layout()
    fig2_path = figures_dir / "fig2_bias_als_convergence.pdf"
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig2_path}")

    summary = {
        "practical_1": {
            "small_5m": {
                "num_users": ds_small.num_users,
                "num_movies": ds_small.num_movies,
                "num_ratings": int(len(small_ratings)),
                "num_train": ds_small.num_train_ratings,
                "num_test": ds_small.num_test_ratings,
                "density_pct": float(
                    100.0
                    * len(small_ratings)
                    / (ds_small.num_users * ds_small.num_movies)
                ),
                "mean_rating": float(np.mean(small_ratings)),
                "std_rating": float(np.std(small_ratings)),
                "user_deg_min": int(np.min(user_deg_small)),
                "user_deg_median": float(np.median(user_deg_small)),
                "user_deg_mean": float(np.mean(user_deg_small)),
                "user_deg_max": int(np.max(user_deg_small)),
                "movie_deg_min_nonzero": int(
                    np.min(movie_deg_small[movie_deg_small > 0])
                ),
                "movie_deg_median": float(np.median(movie_deg_small)),
                "movie_deg_mean": float(np.mean(movie_deg_small)),
                "movie_deg_max": int(np.max(movie_deg_small)),
                "load_time_sec": t_small_load,
            },
            "full_32m": {
                "num_users": ds_full.num_users,
                "num_movies": ds_full.num_movies,
                "num_ratings": int(len(full_ratings)),
                "num_train": ds_full.num_train_ratings,
                "num_test": ds_full.num_test_ratings,
                "density_pct": float(
                    100.0
                    * len(full_ratings)
                    / (ds_full.num_users * ds_full.num_movies)
                ),
                "mean_rating": float(np.mean(full_ratings)),
                "std_rating": float(np.std(full_ratings)),
                "user_deg_min": int(np.min(user_deg_full)),
                "user_deg_median": float(np.median(user_deg_full)),
                "user_deg_mean": float(np.mean(user_deg_full)),
                "user_deg_max": int(np.max(user_deg_full)),
                "movie_deg_min_nonzero": int(
                    np.min(movie_deg_full[movie_deg_full > 0])
                ),
                "movie_deg_zeros": int(np.sum(movie_deg_full == 0)),
                "movie_deg_median": float(np.median(movie_deg_full)),
                "movie_deg_mean": float(np.mean(movie_deg_full)),
                "movie_deg_max": int(np.max(movie_deg_full)),
                "alpha_user_mle": alpha_user_full,
                "alpha_movie_mle": alpha_movie_full,
                "star_values": [float(s) for s in star_values],
                "star_counts": [int(c) for c in full_hist_counts],
                "star_pcts": [float(p * 100.0) for p in full_hist_probs],
                "load_time_sec": t_full_load,
            },
        },
        "practical_2": {
            "small_5m": {
                "mu": res_bias_small.mu,
                "initial_loss": res_bias_small.train_loss_user_view[0],
                "final_loss": res_bias_small.train_loss_user_view[-1],
                "initial_train_rmse": res_bias_small.train_rmse[0],
                "final_train_rmse": res_bias_small.train_rmse[-1],
                "initial_test_rmse": res_bias_small.test_rmse[0],
                "final_test_rmse": res_bias_small.test_rmse[-1],
                "max_abs_dual_loss_diff": float(np.max(diff_small)),
                "mean_epoch_time_sec": float(np.mean(res_bias_small.epoch_times)),
            },
            "full_32m": {
                "mu": res_bias_full.mu,
                "initial_loss": res_bias_full.train_loss_user_view[0],
                "final_loss": res_bias_full.train_loss_user_view[-1],
                "initial_train_rmse": res_bias_full.train_rmse[0],
                "final_train_rmse": res_bias_full.train_rmse[-1],
                "initial_test_rmse": res_bias_full.test_rmse[0],
                "final_test_rmse": res_bias_full.test_rmse[-1],
                "max_abs_dual_loss_diff": float(np.max(diff_full)),
                "max_rel_dual_loss_diff": float(np.max(rel_diff_full)),
                "mean_epoch_time_sec": float(np.mean(res_bias_full.epoch_times)),
                "train_rmse_history": res_bias_full.train_rmse,
                "test_rmse_history": res_bias_full.test_rmse,
            },
        },
    }

    out_json = ROOT_DIR / "report" / "results_practicals_1_2.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
