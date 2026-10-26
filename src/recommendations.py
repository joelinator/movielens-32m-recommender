"""Recommendation utilities, Dummy User inference, and Polarizing Movie analysis.

Implements Practical 4 analytical modules:
1. Closed-form inference of a new ("dummy") user's bias a_new and latent vector
   u_new from a small set of seed ratings {(n_i, r_i)}, followed by top-N
   personalized ranking under different bias-weighting regimes:
   - Full bias: score(n) = u_new^T v_n + b_n
   - Attenuated bias: score(n) = u_new^T v_n + 0.05 * b_n
   - Popularity-filtered pure similarity: score(n) = u_new^T v_n + 0.05 * b_n
     restricted to movies with |Omega_n| >= min_ratings.
2. Polarizing vs. consensus movie analysis via the L2 norm ||v_n||_2 of learned
   item trait vectors, controlling for rating frequency |Omega_n| to separate
   genuine ideological/aesthetic polarization from small-sample noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MovieRecommendation:
    """Single ranked movie recommendation."""

    rank: int
    movie_idx: int
    movie_id: int
    title: str
    genres: str
    score: float
    dot_similarity: float
    movie_bias: float
    rating_count: int


def infer_new_user_profile(
    seed_ratings: Sequence[Tuple[int, float]],
    movie_biases: np.ndarray,
    movie_factors: np.ndarray,
    mu: float,
    lam: float,
    gamma: float,
    tau: float,
    num_alternating_steps: int = 10,
) -> Tuple[float, np.ndarray]:
    """Infer (a_new, u_new) for a new user with seed ratings {(movie_idx, r)}.

    Alternates between the exact closed-form bias update and the K x K ridge
    regression update for u_new while keeping all movie parameters fixed.
    """
    k_dim = movie_factors.shape[1]
    if not seed_ratings:
        return 0.0, np.zeros(k_dim, dtype=np.float32)

    item_indices = np.array([idx for idx, _ in seed_ratings], dtype=np.int32)
    ratings = np.array([r for _, r in seed_ratings], dtype=np.float64)
    v_sub = movie_factors[item_indices].astype(np.float64)
    b_sub = movie_biases[item_indices].astype(np.float64)
    count = len(seed_ratings)

    a_new = 0.0
    u_new = np.zeros(k_dim, dtype=np.float64)
    lhs = lam * (v_sub.T @ v_sub) + tau * np.eye(k_dim, dtype=np.float64)

    for _ in range(num_alternating_steps):
        res_b = ratings - mu - b_sub - (v_sub @ u_new)
        a_new = float((lam * np.sum(res_b)) / (lam * count + gamma))
        res_u = ratings - mu - a_new - b_sub
        rhs = lam * (v_sub.T @ res_u)
        u_new = np.linalg.solve(lhs, rhs)

    return float(a_new), u_new.astype(np.float32)


def rank_movies_for_user(
    u_user: np.ndarray,
    movie_biases: np.ndarray,
    movie_factors: np.ndarray,
    movie_rating_counts: np.ndarray,
    idx_to_movie_id: List[int],
    movie_id_to_title: Dict[int, str],
    movie_id_to_genres: Dict[int, List[str]],
    exclude_movie_indices: Optional[Sequence[int]] = None,
    bias_weight: float = 0.05,
    min_rating_count: int = 0,
    top_k: int = 10,
) -> List[MovieRecommendation]:
    """Rank all candidate movies according to u_user^T v_n + bias_weight * b_n."""
    dots = movie_factors @ u_user
    scores = dots + bias_weight * movie_biases

    mask = movie_rating_counts >= min_rating_count
    if exclude_movie_indices:
        for idx in exclude_movie_indices:
            if 0 <= idx < len(mask):
                mask[idx] = False

    valid_indices = np.where(mask)[0]
    valid_scores = scores[valid_indices]
    top_order = np.argsort(valid_scores)[::-1][:top_k]

    recommendations: List[MovieRecommendation] = []
    for rank_idx, local_pos in enumerate(top_order, start=1):
        m_idx = int(valid_indices[local_pos])
        m_id = int(idx_to_movie_id[m_idx])
        title = movie_id_to_title.get(m_id, f"Movie {m_id}")
        genres = "|".join(movie_id_to_genres.get(m_id, []))
        recommendations.append(
            MovieRecommendation(
                rank=rank_idx,
                movie_idx=m_idx,
                movie_id=m_id,
                title=title,
                genres=genres,
                score=float(scores[m_idx]),
                dot_similarity=float(dots[m_idx]),
                movie_bias=float(movie_biases[m_idx]),
                rating_count=int(movie_rating_counts[m_idx]),
            )
        )
    return recommendations


@dataclass
class PolarizingMovieEntry:
    """Metadata and vector magnitude for a polarizing or consensus movie."""

    rank: int
    movie_idx: int
    movie_id: int
    title: str
    genres: str
    vector_norm: float
    movie_bias: float
    rating_count: int
    rating_std: float


def analyze_polarizing_movies(
    movie_biases: np.ndarray,
    movie_factors: np.ndarray,
    movie_indptr: np.ndarray,
    movie_ratings: np.ndarray,
    idx_to_movie_id: List[int],
    movie_id_to_title: Dict[int, str],
    movie_id_to_genres: Dict[int, List[str]],
    min_ratings: int = 500,
    top_k: int = 10,
) -> Tuple[List[PolarizingMovieEntry], List[PolarizingMovieEntry], np.ndarray]:
    """Identify the most and least polarizing movies via ||v_n||_2.

    Controls for sample size by requiring at least `min_ratings` observations
    so that large norms reflect true audience divergence rather than under-
    regularized small-sample noise.
    """
    norms = np.linalg.norm(movie_factors, axis=1)
    counts = np.diff(movie_indptr)

    # Compute empirical rating standard deviation per movie
    stds = np.zeros(len(counts), dtype=np.float32)
    for n in range(len(counts)):
        if counts[n] >= 2:
            r_slice = movie_ratings[movie_indptr[n] : movie_indptr[n + 1]]
            stds[n] = float(np.std(r_slice))

    valid_mask = counts >= min_ratings
    valid_indices = np.where(valid_mask)[0]
    valid_norms = norms[valid_indices]

    most_order = np.argsort(valid_norms)[::-1][:top_k]
    least_order = np.argsort(valid_norms)[:top_k]

    def _build_list(order: np.ndarray) -> List[PolarizingMovieEntry]:
        entries: List[PolarizingMovieEntry] = []
        for rank_idx, pos in enumerate(order, start=1):
            m_idx = int(valid_indices[pos])
            m_id = int(idx_to_movie_id[m_idx])
            entries.append(
                PolarizingMovieEntry(
                    rank=rank_idx,
                    movie_idx=m_idx,
                    movie_id=m_id,
                    title=movie_id_to_title.get(m_id, f"Movie {m_id}"),
                    genres="|".join(movie_id_to_genres.get(m_id, [])),
                    vector_norm=float(norms[m_idx]),
                    movie_bias=float(movie_biases[m_idx]),
                    rating_count=int(counts[m_idx]),
                    rating_std=float(stds[m_idx]),
                )
            )
        return entries

    return _build_list(most_order), _build_list(least_order), norms
