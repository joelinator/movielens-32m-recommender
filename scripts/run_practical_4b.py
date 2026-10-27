"""Execute Practical 4B experiments on full MovieLens-32M and generate figures.

Generates:
- `report/figures/fig7_movie_2d_embeddings.pdf`: 2D latent movie trait space
  (K=2) trained on MovieLens-32M, highlighting iconic films across genres.
- `report/figures/fig8_polarizing_movies.pdf`: Distribution of movie factor
  norms ||v_n||_2 vs rating frequency |Omega^n| and empirical rating std,
  contrasting the most polarizing and most consensus films.
- `report/results_practical_4b.json`: Top-10 recommendations for the "Lord of
  the Rings" dummy user across the three scoring regimes (beta=1.0, beta=0.05,
  and beta=0.05 with |Omega^n| >= 100), plus top-10 most and least polarizing
  movies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.als_latent import fit_pmf_als
from src.data_indexing import load_indexed_dataset, parse_movies_metadata
from src.recommendations import (
    analyze_polarizing_movies,
    infer_new_user_profile,
    rank_movies_for_user,
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


def load_raw_movie_dicts(
    movies_csv_path: Path,
) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    """Load mapping from integer MovieLens movieId to title and genre list."""
    id_to_title: Dict[int, str] = {}
    id_to_genres: Dict[int, List[str]] = {}
    with movies_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            m_id = int(row[0])
            id_to_title[m_id] = row[1]
            id_to_genres[m_id] = (
                row[2].split("|")
                if row[2] and row[2] != "(no genres listed)"
                else ["(no genres listed)"]
            )
    return id_to_title, id_to_genres


def main() -> None:
    configure_matplotlib()
    figures_dir = ROOT_DIR / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading full MovieLens-32M dataset and movies.csv metadata...")
    ds_full = load_indexed_dataset(ROOT_DIR / "dataset.pkl")
    meta = parse_movies_metadata(
        str(ROOT_DIR / "movies.csv"), ds_full.movies_index
    )
    id_to_title, id_to_genres = load_raw_movie_dicts(ROOT_DIR / "movies.csv")
    idx_to_movie_id = [int(s) for s in ds_full.movies_index_rev]

    movie_total_counts = (
        ds_full.movie_train.row_degrees() + ds_full.movie_test.row_degrees()
    )
    movie_train_counts = ds_full.movie_train.row_degrees()

    # -------------------------------------------------------------------------
    # 1. Train 2D Embedding Model (K=2) on Full MovieLens-32M
    # -------------------------------------------------------------------------
    print("Training 2D PMF model (K=2, 18 epochs) on MovieLens-32M...")
    pmf_k2 = fit_pmf_als(
        ds_full,
        k_dim=2,
        num_iters=18,
        lam=1.0,
        gamma=0.05,
        tau=7.0,
        use_global_mean=True,
        seed=42,
        eval_every=18,
    )
    print(
        f"  K=2 Final Train RMSE={pmf_k2.train_rmse[-1]:.4f}, "
        f"Test RMSE={pmf_k2.test_rmse[-1]:.4f}"
    )

    # Select well-known landmark movies across distinct genres/clusters
    # Find top popular movies and specific iconic titles
    v2d = pmf_k2.movie_factors
    popular_mask = movie_total_counts >= 5000
    pop_indices = np.where(popular_mask)[0]

    # Categorize popular movies by primary genre for color-coding
    cat_colors = {
        "Sci-Fi / Fantasy": "#1f4e78",
        "Drama / Crime": "#1b9e77",
        "Horror / Action": "#d95f02",
        "Family / Comedy": "#7570b3",
    }
    cat_points: Dict[str, List[int]] = {k: [] for k in cat_colors}

    for idx in pop_indices:
        m_id = idx_to_movie_id[idx]
        g_set = set(id_to_genres.get(m_id, []))
        if "Animation" in g_set or "Children" in g_set:
            cat_points["Family / Comedy"].append(int(idx))
        elif "Horror" in g_set:
            cat_points["Horror / Action"].append(int(idx))
        elif "Sci-Fi" in g_set or "Fantasy" in g_set:
            cat_points["Sci-Fi / Fantasy"].append(int(idx))
        elif "Drama" in g_set or "Crime" in g_set:
            cat_points["Drama / Crime"].append(int(idx))
        else:
            cat_points["Family / Comedy"].append(int(idx))

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
    for cat_name, indices_list in cat_points.items():
        pts = v2d[indices_list]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=11,
            alpha=0.42,
            color=cat_colors[cat_name],
            label=cat_name,
            edgecolors="none",
            rasterized=True,
        )

    # Annotate ONLY movies explicitly discussed in Section 4.7 of main.tex
    landmarks_to_plot = [
        (339, "LotR: Fellowship", 1.25, 1.48),
        (613, "LotR: Return", -0.65, 1.75),
        (370, "Inception", 1.15, 2.38),
        (1032, "Twilight", -2.10, 1.85),
        (88, "Titanic", -1.35, 0.35),
        (218, "Godfather", 1.35, 0.72),
        (159, "Pulp Fiction", 2.42, 0.88),
        (26, "Fargo", 2.45, -0.35),
        (168, "Lion King", -1.35, -0.45),
        (415, "Toy Story", 0.85, -0.80),
        (153, "Dumb & Dumber", 0.65, -3.55),
    ]

    annotated_coords = []
    for best_idx, short_label, tx, ty in landmarks_to_plot:
        best_cnt = int(movie_total_counts[best_idx])
        x_val, y_val = float(v2d[best_idx, 0]), float(v2d[best_idx, 1])
        ax.scatter(
            [x_val],
            [y_val],
            s=32,
            color="black",
            marker="o",
            edgecolors="white",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            short_label,
            xy=(x_val, y_val),
            xytext=(tx, ty),
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="white",
                ec="0.55",
                alpha=0.92,
                lw=0.45,
            ),
            arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6, shrinkA=0, shrinkB=2),
            zorder=6,
        )
        annotated_coords.append(
            {
                "label": short_label,
                "movie_idx": best_idx,
                "x": x_val,
                "y": y_val,
                "count": best_cnt,
            }
        )

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlim(-3.0, 3.15)
    ax.set_ylim(-4.05, 3.55)
    ax.set_xlabel("Dim. 1 ($v_{n,1}$)")
    ax.set_ylabel("Dim. 2 ($v_{n,2}$)")
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=7.5, markerscale=1.4)

    fig.tight_layout()
    fig7_path = figures_dir / "fig7_movie_2d_embeddings.pdf"
    fig.savefig(fig7_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig7_path}")

    # -------------------------------------------------------------------------
    # 2. Train High-Capacity Full Model (K=16) on MovieLens-32M
    # -------------------------------------------------------------------------
    print("Training K=16 PMF model (18 epochs) on MovieLens-32M...")
    pmf_k16 = fit_pmf_als(
        ds_full,
        k_dim=16,
        num_iters=18,
        lam=1.0,
        gamma=0.05,
        tau=7.0,
        use_global_mean=True,
        seed=42,
        eval_every=18,
    )
    print(
        f"  K=16 Final Train RMSE={pmf_k16.train_rmse[-1]:.4f}, "
        f"Test RMSE={pmf_k16.test_rmse[-1]:.4f}"
    )

    # Find Lord of the Rings: The Fellowship of the Ring (2001) (movieId=4993)
    lotr_idx = ds_full.movies_index["4993"]
    lotr_title = meta.movie_titles[lotr_idx]
    print(f"Dummy user seed movie: {lotr_title} (internal idx={lotr_idx}, rating=5.0)")

    seed_ratings = [(lotr_idx, 5.0)]
    a_dummy, u_dummy = infer_new_user_profile(
        seed_ratings,
        pmf_k16.movie_biases,
        pmf_k16.movie_factors,
        pmf_k16.mu,
        pmf_k16.lam,
        pmf_k16.gamma,
        pmf_k16.tau,
    )

    recs_full_bias = rank_movies_for_user(
        u_dummy,
        pmf_k16.movie_biases,
        pmf_k16.movie_factors,
        movie_train_counts,
        idx_to_movie_id,
        id_to_title,
        id_to_genres,
        exclude_movie_indices=[lotr_idx],
        bias_weight=1.0,
        min_rating_count=0,
        top_k=10,
    )
    recs_small_bias = rank_movies_for_user(
        u_dummy,
        pmf_k16.movie_biases,
        pmf_k16.movie_factors,
        movie_train_counts,
        idx_to_movie_id,
        id_to_title,
        id_to_genres,
        exclude_movie_indices=[lotr_idx],
        bias_weight=0.05,
        min_rating_count=0,
        top_k=10,
    )
    recs_filtered = rank_movies_for_user(
        u_dummy,
        pmf_k16.movie_biases,
        pmf_k16.movie_factors,
        movie_train_counts,
        idx_to_movie_id,
        id_to_title,
        id_to_genres,
        exclude_movie_indices=[lotr_idx],
        bias_weight=0.05,
        min_rating_count=100,
        top_k=10,
    )

    # Also test a second Dummy User: Animation/Pixar fan (Toy Story = 5.0)
    toystory_idx = ds_full.movies_index["1"]
    _, u_toystory = infer_new_user_profile(
        [(toystory_idx, 5.0)],
        pmf_k16.movie_biases,
        pmf_k16.movie_factors,
        pmf_k16.mu,
        pmf_k16.lam,
        pmf_k16.gamma,
        pmf_k16.tau,
    )
    recs_toystory = rank_movies_for_user(
        u_toystory,
        pmf_k16.movie_biases,
        pmf_k16.movie_factors,
        movie_train_counts,
        idx_to_movie_id,
        id_to_title,
        id_to_genres,
        exclude_movie_indices=[toystory_idx],
        bias_weight=0.05,
        min_rating_count=100,
        top_k=5,
    )

    # -------------------------------------------------------------------------
    # 3. Polarizing vs. Consensus Movies Analysis (||v_n||_2)
    # -------------------------------------------------------------------------
    print("Analyzing Most vs. Least Polarizing Movies (min_ratings=1000)...")
    most_pol, least_pol, all_norms = analyze_polarizing_movies(
        pmf_k16.movie_biases,
        pmf_k16.movie_factors,
        ds_full.movie_train.indptr,
        ds_full.movie_train.ratings,
        idx_to_movie_id,
        id_to_title,
        id_to_genres,
        min_ratings=1000,
        top_k=10,
    )

    # Compute rating std for all movies with >= 100 ratings for scatter plot
    active_mask = movie_train_counts >= 100
    active_indices = np.where(active_mask)[0]
    active_stds = np.zeros(len(active_indices), dtype=np.float32)
    for i, m_idx in enumerate(active_indices):
        s_ptr = ds_full.movie_train.indptr[m_idx]
        e_ptr = ds_full.movie_train.indptr[m_idx + 1]
        active_stds[i] = float(np.std(ds_full.movie_train.ratings[s_ptr:e_ptr]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.75))

    non_empty = movie_train_counts > 0
    ax1.scatter(
        movie_train_counts[non_empty],
        all_norms[non_empty],
        s=4,
        alpha=0.25,
        color="#1f4e78",
        edgecolors="none",
        rasterized=True,
        label="Movies",
    )
    ax1.axvline(
        1000,
        color="#d95f02",
        linestyle="--",
        linewidth=1.2,
        label=r"$|\Omega^n| \geq 10^3$",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"(a) Ratings $|\Omega^n|$")
    ax1.set_ylabel(r"Norm $\|v_n\|_2$")
    ax1.legend(loc="upper left", frameon=False, fontsize=7.5)

    ax2.scatter(
        active_stds,
        all_norms[active_indices],
        s=5,
        alpha=0.25,
        color="#1b9e77",
        edgecolors="none",
        rasterized=True,
        label=r"$|\Omega^n| \geq 100$",
    )
    # Highlight top polarizing and least polarizing on panel (b)
    pol_stds = [e.rating_std for e in most_pol[:5]]
    pol_norms = [e.vector_norm for e in most_pol[:5]]
    con_stds = [e.rating_std for e in least_pol[:5]]
    con_norms = [e.vector_norm for e in least_pol[:5]]
    ax2.scatter(
        pol_stds,
        pol_norms,
        s=28,
        color="#d95f02",
        marker="^",
        zorder=5,
        label="Polarizing",
    )
    ax2.scatter(
        con_stds,
        con_norms,
        s=28,
        color="#7570b3",
        marker="s",
        zorder=5,
        label="Consensus",
    )
    ax2.set_ylim(0, 8.6)
    ax2.set_xlabel(r"(b) Rating Std. $\sigma_n$")
    ax2.set_ylabel(r"Norm $\|v_n\|_2$")
    ax2.legend(loc="upper left", frameon=False, fontsize=7.0, ncol=1)

    fig.tight_layout()
    fig8_path = figures_dir / "fig8_polarizing_movies.pdf"
    fig.savefig(fig8_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig8_path}")

    summary = {
        "k2_model": {
            "train_rmse": pmf_k2.train_rmse[-1],
            "test_rmse": pmf_k2.test_rmse[-1],
            "landmarks": annotated_coords,
        },
        "k16_model": {
            "train_rmse": pmf_k16.train_rmse[-1],
            "test_rmse": pmf_k16.test_rmse[-1],
            "mean_epoch_time_sec": float(np.mean(pmf_k16.epoch_times)),
        },
        "dummy_user_lotr": {
            "seed_title": lotr_title,
            "inferred_bias": a_dummy,
            "inferred_norm": float(np.linalg.norm(u_dummy)),
            "recs_full_bias": [r.__dict__ for r in recs_full_bias],
            "recs_small_bias_unfiltered": [r.__dict__ for r in recs_small_bias],
            "recs_small_bias_filtered_100": [r.__dict__ for r in recs_filtered],
        },
        "dummy_user_toystory": {
            "recs_filtered_100": [r.__dict__ for r in recs_toystory],
        },
        "polarizing_movies_min1000": {
            "most_polarizing": [e.__dict__ for e in most_pol],
            "least_polarizing": [e.__dict__ for e in least_pol],
        },
    }

    out_json = ROOT_DIR / "report" / "results_practical_4b.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
