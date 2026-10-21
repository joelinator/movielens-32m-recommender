"""Execute Practical 5 experiments (Hybrid Genre-Augmented ALS) and figures.

Generates:
- `report/figures/fig9_genre_2d_embeddings.pdf`: 2D latent space of learned
  genre embeddings f_g in R^2 alongside movie genre centroids on MovieLens-32M.
- `report/figures/fig10_genre_als_coldstart.pdf`: Monotonic convergence of
  3-block Genre-ALS and Test RMSE stratified by movie training degree
  |Omega_train^n| (including controlled cold-start movie holdout |Omega_train^n|=0).
- `report/results_practical_5.json`: Exact numerical metrics for Practical 5.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.als_genres import fit_genre_als
from src.als_latent import fit_pmf_als
from src.data_indexing import (
    CSRMatrix,
    IndexedDataset,
    load_indexed_dataset,
    parse_movies_metadata,
)


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


@numba.njit(parallel=True, fastmath=True)
def compute_per_movie_test_sse_and_counts(
    movie_test_indptr: np.ndarray,
    movie_test_indices: np.ndarray,
    movie_test_ratings: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    user_factors: np.ndarray,
    movie_factors: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-movie test SSE and test rating counts over movie_test CSR."""
    num_movies = len(movie_test_indptr) - 1
    k_dim = user_factors.shape[1]
    m_sse = np.zeros(num_movies, dtype=np.float64)
    m_cnt = np.zeros(num_movies, dtype=np.int32)

    for n in numba.prange(num_movies):
        start = movie_test_indptr[n]
        end = movie_test_indptr[n + 1]
        m_cnt[n] = end - start
        b_n = movie_biases[n]
        v_n = movie_factors[n]
        s = 0.0
        for idx in range(start, end):
            m = movie_test_indices[idx]
            u_m = user_factors[m]
            dot_uv = 0.0
            for d in range(k_dim):
                dot_uv += u_m[d] * v_n[d]
            err = movie_test_ratings[idx] - (mu + user_biases[m] + b_n + dot_uv)
            s += err * err
        m_sse[n] = s
    return m_sse, m_cnt


def create_cold_start_holdout_dataset(
    ds: IndexedDataset,
    holdout_movie_indices: np.ndarray,
) -> IndexedDataset:
    """Create a modified dataset where `holdout_movie_indices` have 0 train ratings."""
    holdout_mask_movie = np.zeros(ds.num_movies, dtype=bool)
    holdout_mask_movie[holdout_movie_indices] = True

    # Filter user_train CSR to remove ratings on held-out movies
    u_indptr = ds.user_train.indptr
    u_indices = ds.user_train.indices
    u_ratings = ds.user_train.ratings

    keep_edge = ~holdout_mask_movie[u_indices]
    new_u_indices = np.ascontiguousarray(u_indices[keep_edge])
    new_u_ratings = np.ascontiguousarray(u_ratings[keep_edge])

    # Recompute user_train offsets
    edge_counts = np.add.reduceat(
        keep_edge.astype(np.int32), u_indptr[:-1]
    )
    empty_rows = np.diff(u_indptr) == 0
    edge_counts[empty_rows] = 0
    new_u_offsets = np.zeros(ds.num_users + 1, dtype=np.int32)
    np.cumsum(edge_counts, out=new_u_offsets[1:])

    # Filter movie_train CSR to empty out held-out movie rows
    m_indptr = ds.movie_train.indptr
    m_counts = np.diff(m_indptr).copy()
    m_counts[holdout_movie_indices] = 0
    new_m_offsets = np.zeros(ds.num_movies + 1, dtype=np.int32)
    np.cumsum(m_counts, out=new_m_offsets[1:])

    new_m_indices = np.zeros(int(new_m_offsets[-1]), dtype=np.int32)
    new_m_ratings = np.zeros(int(new_m_offsets[-1]), dtype=np.float32)

    for n in range(ds.num_movies):
        if holdout_mask_movie[n]:
            continue
        s_old, e_old = m_indptr[n], m_indptr[n + 1]
        s_new, e_new = new_m_offsets[n], new_m_offsets[n + 1]
        if e_old > s_old:
            new_m_indices[s_new:e_new] = ds.movie_train.indices[s_old:e_old]
            new_m_ratings[s_new:e_new] = ds.movie_train.ratings[s_old:e_old]

    return IndexedDataset(
        num_users=ds.num_users,
        num_movies=ds.num_movies,
        data_by_user_train=CSRMatrix(
            num_rows=ds.num_users,
            num_cols=ds.num_movies,
            offsets=new_u_offsets,
            indices=new_u_indices,
            ratings=new_u_ratings,
        ),
        data_by_user_test=ds.data_by_user_test,
        data_by_movie_train=CSRMatrix(
            num_rows=ds.num_movies,
            num_cols=ds.num_users,
            offsets=new_m_offsets,
            indices=new_m_indices,
            ratings=new_m_ratings,
        ),
        data_by_movie_test=ds.data_by_movie_test,
        users_index=ds.users_index,
        users_index_rev=ds.users_index_rev,
        movies_index=ds.movies_index,
        movies_index_rev=ds.movies_index_rev,
    )


def main() -> None:
    configure_matplotlib()
    figures_dir = ROOT_DIR / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MovieLens-32M and MovieLens-5M with genre metadata...")
    ds_full = load_indexed_dataset(ROOT_DIR / "dataset.pkl")
    meta_full = parse_movies_metadata(
        str(ROOT_DIR / "movies.csv"), ds_full.movies_index
    )

    genre_counts_per_movie = np.diff(meta_full.movie_genre_offsets).astype(
        np.float32
    )
    genre_weights_full = 1.0 / np.sqrt(np.maximum(1.0, genre_counts_per_movie))
    num_genres = len(meta_full.genre_names)
    print(f"Parsed {num_genres} canonical genres: {meta_full.genre_names}")

    # -------------------------------------------------------------------------
    # 1. Train 2D Genre-Augmented ALS (K=2) on MovieLens-32M
    # -------------------------------------------------------------------------
    print("Training 2D Genre-ALS (K=2, 18 epochs) on MovieLens-32M...")
    genre_k2 = fit_genre_als(
        ds_full,
        meta_full.movie_genre_offsets,
        meta_full.movie_genre_indices,
        genre_weights_full,
        num_genres=num_genres,
        k_dim=2,
        num_iters=18,
        lam=1.0,
        gamma=0.05,
        tau=7.0,
        tau_f=1.0,
        use_global_mean=True,
        seed=42,
    )
    print(
        f"  Genre-ALS (K=2) Train RMSE={genre_k2.train_rmse[-1]:.4f}, "
        f"Test RMSE={genre_k2.test_rmse[-1]:.4f}"
    )

    # Plot Figure 9: 2D Genre Trait Vectors f_g in R^2
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
    f2d = genre_k2.genre_factors

    # Also plot faint background of movie prior centroids v_bar_n
    vbar2d = genre_k2.movie_genre_means
    sample_rng = np.random.default_rng(42)
    sample_idx = sample_rng.choice(
        len(vbar2d), size=min(4000, len(vbar2d)), replace=False
    )
    ax.scatter(
        vbar2d[sample_idx, 0],
        vbar2d[sample_idx, 1],
        s=8,
        alpha=0.18,
        color="#888888",
        edgecolors="none",
        label=r"Movie Genre Prior Centroids $\bar{v}_n$",
        rasterized=True,
    )

    # Group genres by thematic family for visual clarity
    family_map = {
        "Drama": ("Serious / Narrative", "#1b9e77", "o"),
        "Documentary": ("Serious / Narrative", "#1b9e77", "o"),
        "Film-Noir": ("Serious / Narrative", "#1b9e77", "o"),
        "War": ("Serious / Narrative", "#1b9e77", "o"),
        "Crime": ("Serious / Narrative", "#1b9e77", "o"),
        "Mystery": ("Serious / Narrative", "#1b9e77", "o"),
        "Action": ("Action / Speculative / Horror", "#d95f02", "^"),
        "Adventure": ("Action / Speculative / Horror", "#d95f02", "^"),
        "Sci-Fi": ("Action / Speculative / Horror", "#d95f02", "^"),
        "Horror": ("Action / Speculative / Horror", "#d95f02", "^"),
        "Thriller": ("Action / Speculative / Horror", "#d95f02", "^"),
        "Fantasy": ("Family / Light / Musical", "#1f4e78", "s"),
        "Animation": ("Family / Light / Musical", "#1f4e78", "s"),
        "Children": ("Family / Light / Musical", "#1f4e78", "s"),
        "Comedy": ("Family / Light / Musical", "#1f4e78", "s"),
        "Romance": ("Family / Light / Musical", "#1f4e78", "s"),
        "Musical": ("Family / Light / Musical", "#1f4e78", "s"),
        "Western": ("Other / Specialized", "#7570b3", "D"),
        "IMAX": ("Other / Specialized", "#7570b3", "D"),
        "(no genres listed)": ("Other / Specialized", "#7570b3", "D"),
    }

    plotted_families = set()
    genre_coords = []
    for g_id, g_name in enumerate(meta_full.genre_names):
        fam_label, col, mk = family_map.get(
            g_name, ("Other / Specialized", "#7570b3", "D")
        )
        gx, gy = float(f2d[g_id, 0]), float(f2d[g_id, 1])
        lbl = fam_label if fam_label not in plotted_families else None
        if lbl is not None:
            plotted_families.add(fam_label)
        ax.scatter(
            [gx],
            [gy],
            s=52,
            color=col,
            marker=mk,
            edgecolors="black",
            linewidth=0.6,
            zorder=5,
            label=lbl,
        )
        dx = 0.015 if gx >= 0 else -0.015
        dy = 0.015 if g_id % 2 == 0 else -0.022
        ax.annotate(
            g_name,
            xy=(gx, gy),
            xytext=(gx + dx, gy + dy),
            fontsize=8.0,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.14",
                fc="white",
                ec=col,
                alpha=0.88,
                lw=0.5,
            ),
            zorder=6,
        )
        genre_coords.append(
            {
                "genre": g_name,
                "x": gx,
                "y": gy,
                "norm": float(np.linalg.norm(f2d[g_id])),
                "num_movies": int(
                    meta_full.genre_movie_offsets[g_id + 1]
                    - meta_full.genre_movie_offsets[g_id]
                ),
            }
        )

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"Genre Latent Trait Dimension 1 ($f_{g,1}$)")
    ax.set_ylabel(r"Genre Latent Trait Dimension 2 ($f_{g,2}$)")
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig9_path = figures_dir / "fig9_genre_2d_embeddings.pdf"
    fig.savefig(fig9_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig9_path}")

    # -------------------------------------------------------------------------
    # 2. Train K=16 Genre-ALS on Full MovieLens-32M & Degree-Bucket Comparison
    # -------------------------------------------------------------------------
    print("Training K=16 Genre-ALS (18 epochs) on MovieLens-32M...")
    genre_k16 = fit_genre_als(
        ds_full,
        meta_full.movie_genre_offsets,
        meta_full.movie_genre_indices,
        genre_weights_full,
        num_genres=num_genres,
        k_dim=16,
        num_iters=18,
        lam=1.0,
        gamma=0.05,
        tau=7.0,
        tau_f=1.0,
        use_global_mean=True,
        seed=42,
    )
    print("Training K=16 Standard PMF (18 epochs) on MovieLens-32M for comparison...")
    std_k16 = fit_pmf_als(
        ds_full,
        k_dim=16,
        num_iters=18,
        lam=1.0,
        gamma=0.05,
        tau=7.0,
        use_global_mean=True,
        seed=42,
    )

    # Controlled Cold-Start Holdout on MovieLens-5M (holding out 500 movies)
    print("Running Controlled Cold-Start Holdout Experiment on MovieLens-5M...")
    ds_small = load_indexed_dataset(ROOT_DIR / "small_dataset.pkl")
    meta_small = parse_movies_metadata(
        str(ROOT_DIR / "movies.csv"), ds_small.movies_index
    )
    g_cnt_small = np.diff(meta_small.movie_genre_offsets).astype(np.float32)
    g_weights_small = 1.0 / np.sqrt(np.maximum(1.0, g_cnt_small))

    m_test_deg_small = ds_small.movie_test.row_degrees()
    eligible_cold = np.where(m_test_deg_small >= 20)[0]
    rng_cold = np.random.default_rng(123)
    holdout_movies = rng_cold.choice(eligible_cold, size=400, replace=False)

    ds_cold = create_cold_start_holdout_dataset(ds_small, holdout_movies)
    std_cold = fit_pmf_als(
        ds_cold,
        k_dim=12,
        num_iters=15,
        lam=1.0,
        gamma=0.05,
        tau=5.0,
        use_global_mean=True,
        seed=42,
        eval_every=15,
    )
    genre_cold = fit_genre_als(
        ds_cold,
        meta_small.movie_genre_offsets,
        meta_small.movie_genre_indices,
        g_weights_small,
        num_genres=len(meta_small.genre_names),
        k_dim=12,
        num_iters=15,
        lam=1.0,
        gamma=0.05,
        tau=5.0,
        tau_f=0.5,
        use_global_mean=True,
        seed=42,
    )

    # Evaluate RMSE by movie training degree bucket on ds_cold
    m_train_deg_cold = ds_cold.movie_train.row_degrees()
    sse_std_m, cnt_m = compute_per_movie_test_sse_and_counts(
        ds_cold.movie_test.indptr,
        ds_cold.movie_test.indices,
        ds_cold.movie_test.ratings,
        std_cold.user_biases,
        std_cold.movie_biases,
        std_cold.user_factors,
        std_cold.movie_factors,
        std_cold.mu,
    )
    sse_gen_m, _ = compute_per_movie_test_sse_and_counts(
        ds_cold.movie_test.indptr,
        ds_cold.movie_test.indices,
        ds_cold.movie_test.ratings,
        genre_cold.user_biases,
        genre_cold.movie_biases,
        genre_cold.user_factors,
        genre_cold.movie_factors,
        genre_cold.mu,
    )

    m_buckets = [
        ("0 (Cold-Start)", m_train_deg_cold == 0),
        ("1-4", (m_train_deg_cold >= 1) & (m_train_deg_cold <= 4)),
        ("5-19", (m_train_deg_cold >= 5) & (m_train_deg_cold <= 19)),
        ("20-99", (m_train_deg_cold >= 20) & (m_train_deg_cold <= 99)),
        ("100+", m_train_deg_cold >= 100),
    ]
    bucket_names = []
    std_bucket_rmse = []
    gen_bucket_rmse = []
    bucket_test_counts = []

    for b_label, mask in m_buckets:
        tot_cnt = int(np.sum(cnt_m[mask]))
        r_std = float(np.sqrt(np.sum(sse_std_m[mask]) / max(1, tot_cnt)))
        r_gen = float(np.sqrt(np.sum(sse_gen_m[mask]) / max(1, tot_cnt)))
        bucket_names.append(b_label)
        std_bucket_rmse.append(r_std)
        gen_bucket_rmse.append(r_gen)
        bucket_test_counts.append(tot_cnt)
        print(
            f"  Movie Degree [{b_label:14s}] (Test N={tot_cnt:6d}) | "
            f"Std PMF={r_std:.4f} | Genre-ALS={r_gen:.4f} "
            f"(Delta={r_gen - r_std:+.4f})"
        )

    # Also compute bucket RMSEs on the un-held-out full MovieLens-32M (K=16)
    m_train_deg_32m = ds_full.movie_train.row_degrees()
    sse_std_32m, cnt_32m = compute_per_movie_test_sse_and_counts(
        ds_full.movie_test.indptr,
        ds_full.movie_test.indices,
        ds_full.movie_test.ratings,
        std_k16.user_biases,
        std_k16.movie_biases,
        std_k16.user_factors,
        std_k16.movie_factors,
        std_k16.mu,
    )
    sse_gen_32m, _ = compute_per_movie_test_sse_and_counts(
        ds_full.movie_test.indptr,
        ds_full.movie_test.indices,
        ds_full.movie_test.ratings,
        genre_k16.user_biases,
        genre_k16.movie_biases,
        genre_k16.user_factors,
        genre_k16.movie_factors,
        genre_k16.mu,
    )
    buckets_32m_summary = {}
    for b_label, lo, hi in [
        ("0", 0, 1),
        ("1-4", 1, 5),
        ("5-19", 5, 20),
        ("20-99", 20, 100),
        ("100+", 100, 1000000),
    ]:
        mask = (m_train_deg_32m >= lo) & (m_train_deg_32m < hi)
        tc = int(np.sum(cnt_32m[mask]))
        buckets_32m_summary[b_label] = {
            "test_ratings": tc,
            "std_pmf_rmse": float(np.sqrt(np.sum(sse_std_32m[mask]) / max(1, tc))),
            "genre_als_rmse": float(np.sqrt(np.sum(sse_gen_32m[mask]) / max(1, tc))),
        }

    # Plot Figure 10: Genre-ALS Convergence & Cold-Start/Low-Degree Improvement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.75))
    iters = np.arange(1, 19)
    ax1.plot(
        iters,
        np.array(genre_k16.train_loss) / 1e6,
        color="#1f4e78",
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label=r"32M Genre-ALS $\mathcal{L}_{\mathrm{genre}}$",
    )
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        iters,
        genre_k16.test_rmse,
        color="#d95f02",
        linestyle="-.",
        linewidth=1.6,
        label="32M Genre-ALS Test RMSE",
    )
    ax1_twin.plot(
        iters,
        genre_k16.train_rmse,
        color="#1b9e77",
        linestyle="--",
        linewidth=1.4,
        label="32M Genre-ALS Train RMSE",
    )
    ax1_twin.grid(False)
    ax1.set_xlabel("(a) ALS Iteration $t$")
    ax1.set_ylabel(r"Regularized Loss $\mathcal{L}_{\mathrm{genre}}$ ($\times 10^6$)")
    ax1_twin.set_ylabel("32M RMSE (Stars)")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=7.2)

    x_pos = np.arange(len(bucket_names))
    width = 0.36
    ax2.bar(
        x_pos - width / 2,
        std_bucket_rmse,
        width=width,
        color="#7570b3",
        edgecolor="black",
        linewidth=0.4,
        label="Standard PMF (Zero Prior)",
    )
    ax2.bar(
        x_pos + width / 2,
        gen_bucket_rmse,
        width=width,
        color="#1b9e77",
        edgecolor="black",
        linewidth=0.4,
        label=r"Genre-ALS ($\bar{v}_n$ Prior)",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bucket_names, rotation=18)
    ax2.set_ylim(0.75, max(std_bucket_rmse) + 0.04)
    ax2.set_xlabel(r"(b) Movie Training Degree $|\Omega_{\mathrm{train}}^n|$")
    ax2.set_ylabel("Test RMSE (Stars)")
    ax2.legend(loc="upper right", fontsize=7.5, frameon=True)

    fig.tight_layout()
    fig10_path = figures_dir / "fig10_genre_als_coldstart.pdf"
    fig.savefig(fig10_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig10_path}")

    summary = {
        "genre_k2_32m": {
            "train_rmse": genre_k2.train_rmse[-1],
            "test_rmse": genre_k2.test_rmse[-1],
            "genres": genre_coords,
        },
        "genre_k16_32m": {
            "initial_loss": genre_k16.train_loss[0],
            "final_loss": genre_k16.train_loss[-1],
            "train_rmse": genre_k16.train_rmse[-1],
            "test_rmse": genre_k16.test_rmse[-1],
            "mean_epoch_time_sec": float(np.mean(genre_k16.epoch_times)),
            "std_pmf_train_rmse": std_k16.train_rmse[-1],
            "std_pmf_test_rmse": std_k16.test_rmse[-1],
            "buckets_32m": buckets_32m_summary,
        },
        "cold_start_experiment_5m": {
            "bucket_names": bucket_names,
            "bucket_test_counts": bucket_test_counts,
            "std_bucket_rmse": std_bucket_rmse,
            "genre_bucket_rmse": gen_bucket_rmse,
        },
    }

    out_json = ROOT_DIR / "report" / "results_practical_5.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
