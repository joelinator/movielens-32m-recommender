"""Execute Practicals 3 and 4A experiments and generate publication PDF figures.

Generates:
- `report/figures/fig3_pmf_loss_and_rmse.pdf`: PMF (K=10) monotonic regularized
  loss and Train/Test RMSE convergence on MovieLens-5M and MovieLens-32M.
- `report/figures/fig4_hyperparameter_search.pdf`: Grid search over prior
  precisions (tau, gamma) with lambda=1.0.
- `report/figures/fig5_k_scaling_and_overfitting.pdf`: Latent dimension sweep
  K in {0, 2, 4, 8, 10, 16, 20, 32} and Test RMSE by user degree bucket
  comparing K=10 vs K=20.
- `report/figures/fig6_runtime_benchmark.pdf`: Per-epoch execution time and
  parallel speedup across Numba Parallel CSR, Numba Serial CSR, and Batched NumPy.
- `report/results_practicals_3_4a.json`: Numerical results for the report.
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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.als_biases import fit_bias_als
from src.als_latent import (
    compute_per_user_sse_and_counts,
    fit_pmf_als,
)
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
            "legend.fontsize": 8.0,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )


def main() -> None:
    configure_matplotlib()
    figures_dir = ROOT_DIR / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    ds_small = load_indexed_dataset(ROOT_DIR / "small_dataset.pkl")
    ds_full = load_indexed_dataset(ROOT_DIR / "dataset.pkl")

    # Warmup JIT compilation on all 3 backends
    print("Warming up Numba parallel and serial kernels...")
    _ = fit_pmf_als(
        ds_small,
        k_dim=4,
        num_iters=1,
        lam=1.0,
        gamma=0.05,
        tau=5.0,
        backend="numba_parallel",
    )
    _ = fit_pmf_als(
        ds_small,
        k_dim=4,
        num_iters=1,
        lam=1.0,
        gamma=0.05,
        tau=5.0,
        backend="numba_serial",
    )

    # -------------------------------------------------------------------------
    # 1. Practical 3: PMF (K=10) Convergence on 5M and 32M
    # -------------------------------------------------------------------------
    print("Running Practical 3: PMF (K=10) for 20 epochs on 5M and 32M...")
    pmf_small_k10 = fit_pmf_als(
        ds_small,
        k_dim=10,
        num_iters=20,
        lam=1.0,
        gamma=0.05,
        tau=5.0,
        use_global_mean=True,
        seed=42,
    )
    pmf_full_k10 = fit_pmf_als(
        ds_full,
        k_dim=10,
        num_iters=20,
        lam=1.0,
        gamma=0.05,
        tau=7.0,
        use_global_mean=True,
        seed=42,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.75))
    iters = np.arange(1, 21)

    ax1.plot(
        iters,
        np.array(pmf_full_k10.train_loss) / 1e6,
        color="#1f4e78",
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label=r"32M Objective $\mathcal{L}(\Theta)$ ($\tau=7$)",
    )
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        iters,
        np.array(pmf_small_k10.train_loss) / 1e6,
        color="#1b9e77",
        marker="s",
        markersize=3.5,
        linestyle="--",
        linewidth=1.5,
        label=r"5M Objective $\mathcal{L}(\Theta)$ ($\tau=5$)",
    )
    ax1_twin.grid(False)
    ax1.set_xlabel("(a) ALS Iteration $t$")
    ax1.set_ylabel(r"32M Objective $\mathcal{L}(\Theta)$ ($\times 10^6$)")
    ax1_twin.set_ylabel(r"5M Objective $\mathcal{L}(\Theta)$ ($\times 10^6$)")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right")

    ax2.plot(
        iters,
        pmf_full_k10.train_rmse,
        color="#1f4e78",
        linewidth=1.6,
        label="32M Train RMSE ($K=10$)",
    )
    ax2.plot(
        iters,
        pmf_full_k10.test_rmse,
        color="#d95f02",
        linewidth=1.6,
        linestyle="-.",
        label="32M Test RMSE ($K=10$)",
    )
    ax2.plot(
        iters,
        pmf_small_k10.train_rmse,
        color="#1b9e77",
        linewidth=1.4,
        linestyle="--",
        label="5M Train RMSE ($K=10$)",
    )
    ax2.plot(
        iters,
        pmf_small_k10.test_rmse,
        color="#7570b3",
        linewidth=1.4,
        linestyle=":",
        label="5M Test RMSE ($K=10$)",
    )
    ax2.set_xlabel("(b) ALS Iteration $t$")
    ax2.set_ylabel("Root Mean Squared Error (Stars)")
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig3_path = figures_dir / "fig3_pmf_loss_and_rmse.pdf"
    fig.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig3_path}")

    # -------------------------------------------------------------------------
    # 2. Practical 4A: Hyperparameter Search over (tau, gamma) with lambda=1.0
    # -------------------------------------------------------------------------
    print("Running Hyperparameter Grid Search on MovieLens-5M (K=10, 12 epochs)...")
    tau_grid = [0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0]
    gamma_grid = [0.01, 0.05, 0.25, 1.0]
    hp_results = {}

    for gamma_val in gamma_grid:
        hp_results[str(gamma_val)] = {"train_rmse": [], "test_rmse": []}
        for tau_val in tau_grid:
            res = fit_pmf_als(
                ds_small,
                k_dim=10,
                num_iters=12,
                lam=1.0,
                gamma=gamma_val,
                tau=tau_val,
                use_global_mean=True,
                seed=42,
                eval_every=12,
            )
            hp_results[str(gamma_val)]["train_rmse"].append(res.train_rmse[-1])
            hp_results[str(gamma_val)]["test_rmse"].append(res.test_rmse[-1])
            print(
                f"  gamma={gamma_val}, tau={tau_val} -> "
                f"Train={res.train_rmse[-1]:.4f}, Test={res.test_rmse[-1]:.4f}"
            )

    # Also test tau sweep on full 32M dataset (K=10, gamma=0.05)
    tau_grid_32m = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0]
    hp_32m_train = []
    hp_32m_test = []
    print("Running tau sensitivity sweep on MovieLens-32M (K=10, 12 epochs)...")
    for tau_val in tau_grid_32m:
        res = fit_pmf_als(
            ds_full,
            k_dim=10,
            num_iters=12,
            lam=1.0,
            gamma=0.05,
            tau=tau_val,
            use_global_mean=True,
            seed=42,
            eval_every=12,
        )
        hp_32m_train.append(res.train_rmse[-1])
        hp_32m_test.append(res.test_rmse[-1])
        print(
            f"  [32M] tau={tau_val} -> "
            f"Train={res.train_rmse[-1]:.4f}, Test={res.test_rmse[-1]:.4f}"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.75))
    colors = ["#1f4e78", "#d95f02", "#1b9e77", "#7570b3"]
    markers = ["o", "s", "^", "D"]
    for idx, gamma_val in enumerate(gamma_grid):
        ax1.plot(
            tau_grid,
            hp_results[str(gamma_val)]["test_rmse"],
            color=colors[idx],
            marker=markers[idx],
            markersize=4,
            linewidth=1.4,
            label=rf"5M Test ($\gamma={gamma_val}$)",
        )
    ax1.set_xlabel(r"(a) Factor Prior Precision $\tau$ ($\lambda=1.0$)")
    ax1.set_ylabel("5M Test RMSE (Stars)")
    ax1.legend(loc="upper right", frameon=True)

    ax2.plot(
        tau_grid_32m,
        hp_32m_test,
        color="#d95f02",
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        label=r"32M Test RMSE ($\gamma=0.05$)",
    )
    ax2.plot(
        tau_grid_32m,
        hp_32m_train,
        color="#1f4e78",
        marker="s",
        markersize=4.5,
        linestyle="--",
        linewidth=1.5,
        label=r"32M Train RMSE ($\gamma=0.05$)",
    )
    ax2.set_xlabel(r"(b) Factor Prior Precision $\tau$ on MovieLens-32M")
    ax2.set_ylabel("32M RMSE (Stars)")
    ax2.legend(loc="center right", frameon=True)

    fig.tight_layout()
    fig4_path = figures_dir / "fig4_hyperparameter_search.pdf"
    fig.savefig(fig4_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig4_path}")



if __name__ == '__main__':
    main()
