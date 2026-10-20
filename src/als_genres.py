"""Hybrid Genre-Augmented Matrix Factorization via ALS (Practical 5).

Incorporates movie genre side information from `movies.csv` into the item
prior distribution (feature-informed Gaussian prior):
    v_n ~ N(v_bar_n, tau^{-1} I_K),   where v_bar_n = (1 / sqrt(|F(n)|)) sum_{g in F(n)} f_g

The regularized negative log-posterior objective is:
    L(a, b, U, V, F) = (lambda / 2) sum_{(m,n) in Omega} (r_{m,n} - mu - a_m - b_n - u_m^T v_n)^2
                       + (gamma / 2) (||a||_2^2 + ||b||_2^2)
                       + (tau / 2) sum_{m=1}^M ||u_m||_2^2
                       + (tau / 2) sum_{n=1}^N ||v_n - (1 / sqrt(|F(n)|)) sum_{g in F(n)} f_g||_2^2
                       + (tau_f / 2) sum_{g=1}^G ||f_g||_2^2

This yields a 3-block Alternating Least Squares algorithm:
1. Update user parameters (a_m, u_m) in parallel across M users.
2. Update movie parameters (b_n, v_n) in parallel across N movies, pulled
   toward their genre prior mean v_bar_n ( enabling non-zero embeddings even
   for cold-start movies with |Omega_n| = 0!).
3. Update genre latent vectors {f_g}_{g=1}^G jointly by solving the G x G
   coupled linear system across all N movies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List, Tuple

import numba
import numpy as np

from src.als_latent import (
    _solve_spd_cholesky_inplace,
    compute_pmf_sse_numba,
    update_biases_and_factors_numba,
)
from src.data_indexing import IndexedDataset


@numba.njit(parallel=True, fastmath=True)
def compute_movie_genre_means(
    genre_indptr: np.ndarray,
    genre_indices: np.ndarray,
    genre_weights: np.ndarray,
    genre_factors: np.ndarray,
    movie_genre_means: np.ndarray,
) -> None:
    """Compute v_bar_n = sum_{g in F(n)} w_n * f_g for each movie n."""
    num_movies = len(genre_indptr) - 1
    k_dim = genre_factors.shape[1]
    for n in numba.prange(num_movies):
        start = genre_indptr[n]
        end = genre_indptr[n + 1]
        w = genre_weights[n]
        for d in range(k_dim):
            s = 0.0
            for idx in range(start, end):
                g = genre_indices[idx]
                s += genre_factors[g, d]
            movie_genre_means[n, d] = w * s


@numba.njit(parallel=True, fastmath=True)
def update_movies_with_genre_prior_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    ratings: np.ndarray,
    movie_biases: np.ndarray,
    user_biases: np.ndarray,
    movie_factors: np.ndarray,
    user_factors: np.ndarray,
    movie_genre_means: np.ndarray,
    mu: float,
    lam: float,
    gamma: float,
    tau: float,
) -> None:
    """Update movie biases b_n and factors v_n with genre prior mean v_bar_n.

    Normal equations for v_n:
        (lam * sum_{m in Omega_n} u_m u_m^T + tau * I_K) v_n
            = lam * sum_{m in Omega_n} (r_{m,n} - mu - a_m - b_n) u_m + tau * v_bar_n

    Notice that when a movie has zero training ratings (|Omega_n| = 0),
    its optimal factor automatically reduces to v_n = v_bar_n!
    """
    num_movies = len(indptr) - 1
    k_dim = movie_factors.shape[1]

    for n in numba.prange(num_movies):
        start = indptr[n]
        end = indptr[n + 1]
        count = end - start
        if count == 0:
            movie_biases[n] = 0.0
            for d in range(k_dim):
                movie_factors[n, d] = movie_genre_means[n, d]
            continue

        # 1. Update movie bias b_n
        res_bias = 0.0
        v_n = movie_factors[n]
        for idx in range(start, end):
            m = indices[idx]
            u_m = user_factors[m]
            dot_uv = 0.0
            for d in range(k_dim):
                dot_uv += u_m[d] * v_n[d]
            res_bias += ratings[idx] - mu - user_biases[m] - dot_uv

        new_bias = (lam * res_bias) / (lam * count + gamma)
        movie_biases[n] = new_bias

        # 2. Update movie factor v_n pulled toward v_bar_n
        a_mat = np.zeros((k_dim, k_dim), dtype=np.float64)
        b_vec = np.zeros(k_dim, dtype=np.float64)
        for d in range(k_dim):
            a_mat[d, d] = tau
            b_vec[d] = tau * float(movie_genre_means[n, d])

        for idx in range(start, end):
            m = indices[idx]
            u_m = user_factors[m]
            r_tilde = ratings[idx] - mu - user_biases[m] - new_bias
            lam_r = lam * r_tilde
            for d1 in range(k_dim):
                u_d1 = float(u_m[d1])
                b_vec[d1] += lam_r * u_d1
                lam_u_d1 = lam * u_d1
                for d2 in range(d1 + 1):
                    a_mat[d1, d2] += lam_u_d1 * float(u_m[d2])

        _solve_spd_cholesky_inplace(a_mat, b_vec)
        for d in range(k_dim):
            movie_factors[n, d] = np.float32(b_vec[d])


def update_genre_factors(
    genre_indptr: np.ndarray,
    genre_indices: np.ndarray,
    genre_weights: np.ndarray,
    movie_factors: np.ndarray,
    num_genres: int,
    tau: float,
    tau_f: float,
) -> np.ndarray:
    """Solve for the G x K genre embedding matrix F in closed form.

    Minimizing (tau / 2) ||V - W F||_F^2 + (tau_f / 2) ||F||_F^2 where
    W_{n,g} = 1 / sqrt(|F(n)|) if g in F(n) else 0 yields:
        (tau * W^T W + tau_f * I_G) F = tau * W^T V
    Since G <= 20, this G x G linear system is tiny and solved exactly.
    """
    num_movies = len(genre_indptr) - 1
    # Construct dense N x G incidence matrix W (since G = 20, N x 20 is only 6.7 MB)
    w_mat = np.zeros((num_movies, num_genres), dtype=np.float64)
    for n in range(num_movies):
        start = genre_indptr[n]
        end = genre_indptr[n + 1]
        if end > start:
            w_mat[n, genre_indices[start:end]] = float(genre_weights[n])

    lhs = tau * (w_mat.T @ w_mat) + tau_f * np.eye(num_genres, dtype=np.float64)
    rhs = tau * (w_mat.T @ movie_factors.astype(np.float64))
    solved_f = np.linalg.solve(lhs, rhs)
    return solved_f.astype(np.float32)


def compute_genre_pmf_objective(
    sse: float,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    user_factors: np.ndarray,
    movie_factors: np.ndarray,
    movie_genre_means: np.ndarray,
    genre_factors: np.ndarray,
    lam: float,
    gamma: float,
    tau: float,
    tau_f: float,
) -> float:
    """Compute exact regularized loss for the Genre-Augmented PMF model."""
    reg_biases = float(
        np.dot(user_biases, user_biases) + np.dot(movie_biases, movie_biases)
    )
    reg_u = float(np.sum(user_factors.astype(np.float64) ** 2))
    diff_v = movie_factors.astype(np.float64) - movie_genre_means.astype(
        np.float64
    )
    reg_v = float(np.sum(diff_v**2))
    reg_f = float(np.sum(genre_factors.astype(np.float64) ** 2))
    return (
        0.5 * lam * sse
        + 0.5 * gamma * reg_biases
        + 0.5 * tau * (reg_u + reg_v)
        + 0.5 * tau_f * reg_f
    )


@dataclass
class GenreALSResult:
    """Training history and learned parameters for Hybrid Genre-ALS."""

    user_biases: np.ndarray
    movie_biases: np.ndarray
    user_factors: np.ndarray
    movie_factors: np.ndarray
    genre_factors: np.ndarray
    movie_genre_means: np.ndarray
    mu: float
    k_dim: int
    lam: float
    gamma: float
    tau: float
    tau_f: float
    train_loss: List[float] = field(default_factory=list)
    train_rmse: List[float] = field(default_factory=list)
    test_rmse: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)

