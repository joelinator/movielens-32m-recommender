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
        markersize=3.2,
        linewidth=1.5,
        label=r"32M ($\tau=7$)",
    )
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        iters,
        np.array(pmf_small_k10.train_loss) / 1e6,
        color="#1b9e77",
        marker="s",
        markersize=3.2,
        linestyle="--",
        linewidth=1.5,
        label=r"5M ($\tau=5$)",
    )
    ax1_twin.grid(False)
    ax1.set_xlabel("(a) Iter. $t$")
    ax1.set_ylabel(r"32M Loss ($\times 10^6$)")
    ax1_twin.set_ylabel(r"5M Loss ($\times 10^6$)")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", frameon=False)

    ax2.plot(
        iters,
        pmf_full_k10.train_rmse,
        color="#1f4e78",
        linewidth=1.5,
        label="32M Train",
    )
    ax2.plot(
        iters,
        pmf_full_k10.test_rmse,
        color="#d95f02",
        linewidth=1.5,
        linestyle="-.",
        label="32M Test",
    )
    ax2.plot(
        iters,
        pmf_small_k10.train_rmse,
        color="#1b9e77",
        linewidth=1.4,
        linestyle="--",
        label="5M Train",
    )
    ax2.plot(
        iters,
        pmf_small_k10.test_rmse,
        color="#7570b3",
        linewidth=1.4,
        linestyle=":",
        label="5M Test",
    )
    ax2.set_xlabel("(b) Iter. $t$")
    ax2.set_ylabel("RMSE")
    ax2.legend(loc="upper right", fontsize=7.0, ncol=2, frameon=False)

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
            markersize=3.8,
            linewidth=1.4,
            label=rf"$\gamma={gamma_val}$",
        )
    ax1.set_xlabel(r"(a) Reg. $\tau$ (5M)")
    ax1.set_ylabel("Test RMSE")
    ax1.legend(loc="upper right", fontsize=7.0, ncol=2, frameon=False)

    ax2.plot(
        tau_grid_32m,
        hp_32m_test,
        color="#d95f02",
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        label="Test",
    )
    ax2.plot(
        tau_grid_32m,
        hp_32m_train,
        color="#1f4e78",
        marker="s",
        markersize=4.0,
        linestyle="--",
        linewidth=1.5,
        label="Train",
    )
    ax2.set_xlabel(r"(b) Reg. $\tau$ (32M)")
    ax2.set_ylabel("RMSE")
    ax2.legend(loc="center right", frameon=False)

    fig.tight_layout()
    fig4_path = figures_dir / "fig4_hyperparameter_search.pdf"
    fig.savefig(fig4_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig4_path}")

    # -------------------------------------------------------------------------
    # 3. Practical 4A: Latent Dimension K Scaling & Overfitting Diagnostics
    # -------------------------------------------------------------------------
    print("Running Latent Dimension K Sweep on MovieLens-5M and MovieLens-32M...")
    k_list_5m = [0, 2, 4, 8, 10, 16, 20, 32]
    k_5m_tuned_train, k_5m_tuned_test = [], []
    k_5m_weak_train, k_5m_weak_test = [], []

    # K=0 is Bias-Only ALS
    bias_5m = fit_bias_als(
        ds_small,
        num_iters=15,
        lam=1.0,
        gamma=0.05,
        use_global_mean=True,
        verify_dual_views=False,
    )
    k_5m_tuned_train.append(bias_5m.train_rmse[-1])
    k_5m_tuned_test.append(bias_5m.test_rmse[-1])
    k_5m_weak_train.append(bias_5m.train_rmse[-1])
    k_5m_weak_test.append(bias_5m.test_rmse[-1])

    models_5m_for_buckets = {}
    for k_val in k_list_5m[1:]:
        res_tuned = fit_pmf_als(
            ds_small,
            k_dim=k_val,
            num_iters=15,
            lam=1.0,
            gamma=0.05,
            tau=5.0,
            use_global_mean=True,
            seed=42,
            eval_every=15,
        )
        res_weak = fit_pmf_als(
            ds_small,
            k_dim=k_val,
            num_iters=15,
            lam=1.0,
            gamma=0.05,
            tau=1.0,
            use_global_mean=True,
            seed=42,
            eval_every=15,
        )
        k_5m_tuned_train.append(res_tuned.train_rmse[-1])
        k_5m_tuned_test.append(res_tuned.test_rmse[-1])
        k_5m_weak_train.append(res_weak.train_rmse[-1])
        k_5m_weak_test.append(res_weak.test_rmse[-1])
        if k_val in (10, 20):
            models_5m_for_buckets[f"K{k_val}_tuned"] = res_tuned
            models_5m_for_buckets[f"K{k_val}_weak"] = res_weak
        print(
            f"  [5M] K={k_val:2d} | Tuned(tau=5) Test={res_tuned.test_rmse[-1]:.4f} "
            f"| Weak(tau=1) Test={res_weak.test_rmse[-1]:.4f}"
        )

    # Also run K in {0, 2, 4, 8, 10, 16, 20} on full MovieLens-32M
    k_list_32m = [0, 2, 4, 8, 10, 16, 20]
    k_32m_train, k_32m_test = [], []
    bias_32m = fit_bias_als(
        ds_full,
        num_iters=15,
        lam=1.0,
        gamma=0.05,
        use_global_mean=True,
        verify_dual_views=False,
    )
    k_32m_train.append(bias_32m.train_rmse[-1])
    k_32m_test.append(bias_32m.test_rmse[-1])

    models_32m_for_buckets = {}
    for k_val in k_list_32m[1:]:
        res_32m = fit_pmf_als(
            ds_full,
            k_dim=k_val,
            num_iters=15,
            lam=1.0,
            gamma=0.05,
            tau=7.0,
            use_global_mean=True,
            seed=42,
            eval_every=15,
        )
        k_32m_train.append(res_32m.train_rmse[-1])
        k_32m_test.append(res_32m.test_rmse[-1])
        if k_val in (10, 20):
            models_32m_for_buckets[f"K{k_val}_tuned"] = res_32m
        print(
            f"  [32M] K={k_val:2d} | Train={res_32m.train_rmse[-1]:.4f} "
            f"| Test={res_32m.test_rmse[-1]:.4f} "
            f"| MeanEpoch={np.mean(res_32m.epoch_times):.2f}s"
        )

    # Evaluate degree-stratified Test RMSE for K=10 vs K=20 on 5M (tau=1 and tau=5)
    user_train_deg_5m = ds_small.user_train.row_degrees()
    bucket_edges = [0, 20, 40, 80, 160, 320, 100000]
    bucket_labels = ["<20", "20-39", "40-79", "80-159", "160-319", "320+"]
    bucket_rmse = {}

    for name, model in models_5m_for_buckets.items():
        u_sse, u_cnt = compute_per_user_sse_and_counts(
            ds_small.user_test.indptr,
            ds_small.user_test.indices,
            ds_small.user_test.ratings,
            model.user_biases,
            model.movie_biases,
            model.user_factors,
            model.movie_factors,
            model.mu,
        )
        rmses = []
        for b_idx in range(len(bucket_labels)):
            lo, hi = bucket_edges[b_idx], bucket_edges[b_idx + 1]
            mask = (user_train_deg_5m >= lo) & (user_train_deg_5m < hi)
            tot_sse = float(np.sum(u_sse[mask]))
            tot_cnt = int(np.sum(u_cnt[mask]))
            rmses.append(float(np.sqrt(tot_sse / max(1, tot_cnt))))
        bucket_rmse[name] = rmses

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.55))
    ax1.plot(
        k_list_32m,
        k_32m_test,
        color="#1f4e78",
        marker="o",
        markersize=3.8,
        linewidth=1.5,
        label="32M Test",
    )
    ax1.plot(
        k_list_32m,
        k_32m_train,
        color="#1f4e78",
        marker="o",
        markersize=3.2,
        linestyle="--",
        linewidth=1.2,
        label="32M Train",
    )
    ax1.plot(
        k_list_5m,
        k_5m_tuned_test,
        color="#1b9e77",
        marker="s",
        markersize=3.8,
        linewidth=1.4,
        label=r"5M ($\tau=5$)",
    )
    ax1.plot(
        k_list_5m,
        k_5m_weak_test,
        color="#d95f02",
        marker="^",
        markersize=3.8,
        linestyle="-.",
        linewidth=1.4,
        label=r"5M ($\tau=1$)",
    )
    ax1.set_xlabel("(a) Rank $K$")
    ax1.set_ylabel("RMSE")
    ax1.legend(loc="upper right", fontsize=7.0, ncol=2, frameon=False)

    x_b = np.arange(len(bucket_labels))
    ax2.plot(
        x_b,
        bucket_rmse["K10_weak"],
        color="#1f4e78",
        marker="o",
        markersize=3.8,
        linestyle="--",
        linewidth=1.4,
        label=r"$K=10, \tau=1$",
    )
    ax2.plot(
        x_b,
        bucket_rmse["K20_weak"],
        color="#d95f02",
        marker="^",
        markersize=3.8,
        linestyle="--",
        linewidth=1.4,
        label=r"$K=20, \tau=1$",
    )
    ax2.plot(
        x_b,
        bucket_rmse["K10_tuned"],
        color="#1b9e77",
        marker="s",
        markersize=3.8,
        linewidth=1.4,
        label=r"$K=10, \tau=5$",
    )
    ax2.plot(
        x_b,
        bucket_rmse["K20_tuned"],
        color="#7570b3",
        marker="D",
        markersize=3.8,
        linewidth=1.4,
        label=r"$K=20, \tau=5$",
    )
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(bucket_labels, fontsize=7.5)
    ax2.set_xlabel(r"(b) User Deg. $|\Omega_m|$")
    ax2.set_ylabel("Test RMSE")
    ax2.legend(loc="upper right", fontsize=7.0, ncol=2, frameon=False)

    fig.tight_layout()
    fig5_path = figures_dir / "fig5_k_scaling_and_overfitting.pdf"
    fig.savefig(fig5_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig5_path}")

    # -------------------------------------------------------------------------
    # 4. Practical 4 Bonus: Runtime & Parallel Speedup Benchmark
    # -------------------------------------------------------------------------
    print("Running Runtime & Parallel Speedup Benchmarks...")
    bench_k = [2, 4, 8, 10, 16, 20]
    times_5m_parallel = []
    times_5m_serial = []
    times_5m_numpy = []
    times_32m_parallel = []

    for k_val in bench_k:
        r_par = fit_pmf_als(
            ds_small,
            k_dim=k_val,
            num_iters=3,
            backend="numba_parallel",
            eval_every=10,
        )
        t_par = float(np.median(r_par.epoch_times))
        times_5m_parallel.append(t_par)

        r_ser = fit_pmf_als(
            ds_small,
            k_dim=k_val,
            num_iters=2,
            backend="numba_serial",
            eval_every=10,
        )
        t_ser = float(np.median(r_ser.epoch_times))
        times_5m_serial.append(t_ser)

        # Run 1 epoch of Batched NumPy on 5M for K <= 10 (above K=10 outer-product memory explodes)
        if k_val <= 10:
            r_np = fit_pmf_als(
                ds_small,
                k_dim=k_val,
                num_iters=1,
                backend="numpy_batched",
                eval_every=10,
            )
            t_np = float(r_np.epoch_times[0])
        else:
            t_np = float("nan")
        times_5m_numpy.append(t_np)

        # Full 32M parallel epoch time
        r_32m = fit_pmf_als(
            ds_full,
            k_dim=k_val,
            num_iters=3,
            backend="numba_parallel",
            eval_every=10,
        )
        t_32m = float(np.median(r_32m.epoch_times))
        times_32m_parallel.append(t_32m)

        print(
            f"  K={k_val:2d} | 5M Par={t_par:.3f}s, Ser={t_ser:.3f}s "
            f"(Speedup={t_ser/t_par:.2f}x), NumPy={t_np:.2f}s | 32M Par={t_32m:.3f}s"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.55))
    valid_np_k = [k for k, t in zip(bench_k, times_5m_numpy) if not np.isnan(t)]
    valid_np_t = [t for t in times_5m_numpy if not np.isnan(t)]

    ax1.plot(
        valid_np_k,
        valid_np_t,
        color="#d95f02",
        marker="^",
        markersize=4.0,
        linewidth=1.4,
        label="5M NumPy",
    )
    ax1.plot(
        bench_k,
        times_5m_serial,
        color="#7570b3",
        marker="s",
        markersize=4.0,
        linewidth=1.4,
        label="5M 1T",
    )
    ax1.plot(
        bench_k,
        times_5m_parallel,
        color="#1b9e77",
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        label="5M 12T",
    )
    ax1.plot(
        bench_k,
        times_32m_parallel,
        color="#1f4e78",
        marker="D",
        markersize=4.0,
        linestyle="--",
        linewidth=1.5,
        label="32M 12T",
    )
    ax1.set_yscale("log")
    ax1.set_ylim(0.035, 35.0)
    ax1.set_xlabel("(a) Rank $K$")
    ax1.set_ylabel("Time / Ep. (s)")
    ax1.legend(loc="upper left", fontsize=7.0, ncol=2, frameon=False)

    speedup_vs_serial = np.array(times_5m_serial) / np.array(times_5m_parallel)
    speedup_vs_numpy = np.array(valid_np_t) / np.array(
        times_5m_parallel[: len(valid_np_t)]
    )

    ax2.plot(
        valid_np_k,
        speedup_vs_numpy,
        color="#d95f02",
        marker="^",
        markersize=4.0,
        linewidth=1.5,
        label="vs. NumPy",
    )
    ax2.plot(
        bench_k,
        speedup_vs_serial,
        color="#1f4e78",
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        label="12T vs. 1T",
    )
    ax2.set_xlabel("(b) Rank $K$")
    ax2.set_ylabel("Speedup ($\\times$)")
    ax2.legend(loc="center right", frameon=False)

    fig.tight_layout()
    fig6_path = figures_dir / "fig6_runtime_benchmark.pdf"
    fig.savefig(fig6_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig6_path}")

    summary = {
        "practical_3": {
            "small_5m_k10": {
                "initial_loss": pmf_small_k10.train_loss[0],
                "final_loss": pmf_small_k10.train_loss[-1],
                "initial_train_rmse": pmf_small_k10.train_rmse[0],
                "final_train_rmse": pmf_small_k10.train_rmse[-1],
                "initial_test_rmse": pmf_small_k10.test_rmse[0],
                "final_test_rmse": pmf_small_k10.test_rmse[-1],
                "mean_epoch_time_sec": float(np.mean(pmf_small_k10.epoch_times)),
            },
            "full_32m_k10": {
                "initial_loss": pmf_full_k10.train_loss[0],
                "final_loss": pmf_full_k10.train_loss[-1],
                "initial_train_rmse": pmf_full_k10.train_rmse[0],
                "final_train_rmse": pmf_full_k10.train_rmse[-1],
                "initial_test_rmse": pmf_full_k10.test_rmse[0],
                "final_test_rmse": pmf_full_k10.test_rmse[-1],
                "mean_epoch_time_sec": float(np.mean(pmf_full_k10.epoch_times)),
            },
        },
        "practical_4a": {
            "tau_grid_5m": tau_grid,
            "gamma_grid_5m": gamma_grid,
            "hp_results_5m": hp_results,
            "tau_grid_32m": tau_grid_32m,
            "hp_32m_train": hp_32m_train,
            "hp_32m_test": hp_32m_test,
            "k_list_5m": k_list_5m,
            "k_5m_tuned_train": k_5m_tuned_train,
            "k_5m_tuned_test": k_5m_tuned_test,
            "k_5m_weak_train": k_5m_weak_train,
            "k_5m_weak_test": k_5m_weak_test,
            "k_list_32m": k_list_32m,
            "k_32m_train": k_32m_train,
            "k_32m_test": k_32m_test,
            "bucket_labels": bucket_labels,
            "bucket_rmse": bucket_rmse,
            "bench_k": bench_k,
            "times_5m_parallel": times_5m_parallel,
            "times_5m_serial": times_5m_serial,
            "times_5m_numpy": times_5m_numpy,
            "times_32m_parallel": times_32m_parallel,
        },
    }

    out_json = ROOT_DIR / "report" / "results_practicals_3_4a.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
