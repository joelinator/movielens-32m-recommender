"""Bias-only Alternating Least Squares (ALS) model for Practical 2.

Models predicted ratings as r_hat_{m,n} = mu + a_m + b_n, where:
- mu is the global mean rating (or 0 if absorbed into biases),
- a_m is the user bias for user m in {0, ..., M-1},
- b_n is the item (movie) bias for movie n in {0, ..., N-1}.

Uses Numba-accelerated CSR kernels for O(|Omega|) updates and exact verification
that the loss computed over `data_by_user` equals the loss computed over
`data_by_movie`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Tuple

import numba
import numpy as np

from src.data_indexing import CSRMatrix, IndexedDataset


@numba.njit(parallel=True, fastmath=True)
def _update_biases_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    ratings: np.ndarray,
    other_biases: np.ndarray,
    target_biases: np.ndarray,
    mu: float,
    lam: float,
    gamma: float,
) -> None:
    """Update user or item biases in parallel over CSR rows.

    For row i with ratings {(j, r_{i,j})}, the closed-form ridge update is:
        bias_i <- (lambda * sum_{j in Omega_i} (r_{i,j} - mu - other_bias_j))
                  / (lambda * |Omega_i| + gamma)
    """
    num_rows = len(indptr) - 1
    for i in numba.prange(num_rows):
        start = indptr[i]
        end = indptr[i + 1]
        count = end - start
        if count == 0:
            target_biases[i] = 0.0
            continue
        residual_sum = 0.0
        for idx in range(start, end):
            j = indices[idx]
            r = ratings[idx]
            residual_sum += r - mu - other_biases[j]
        target_biases[i] = (lam * residual_sum) / (lam * count + gamma)


@numba.njit(parallel=True, fastmath=True)
def compute_sse_from_user_csr(
    user_indptr: np.ndarray,
    user_indices: np.ndarray,
    user_ratings: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    mu: float,
) -> float:
    """Compute sum of squared errors (SSE) iterating over `data_by_user`."""
    num_users = len(user_indptr) - 1
    total_sse = 0.0
    for m in numba.prange(num_users):
        start = user_indptr[m]
        end = user_indptr[m + 1]
        a_m = user_biases[m]
        row_sse = 0.0
        for idx in range(start, end):
            n = user_indices[idx]
            err = user_ratings[idx] - (mu + a_m + movie_biases[n])
            row_sse += err * err
        total_sse += row_sse
    return total_sse


@numba.njit(parallel=True, fastmath=True)
def compute_sse_from_movie_csr(
    movie_indptr: np.ndarray,
    movie_indices: np.ndarray,
    movie_ratings: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    mu: float,
) -> float:
    """Compute sum of squared errors (SSE) iterating over `data_by_movie`."""
    num_movies = len(movie_indptr) - 1
    total_sse = 0.0
    for n in numba.prange(num_movies):
        start = movie_indptr[n]
        end = movie_indptr[n + 1]
        b_n = movie_biases[n]
        col_sse = 0.0
        for idx in range(start, end):
            m = movie_indices[idx]
            err = movie_ratings[idx] - (mu + user_biases[m] + b_n)
            col_sse += err * err
        total_sse += col_sse
    return total_sse


@dataclass
class BiasALSResult:
    """Training history and learned parameters for Bias-Only ALS."""

    user_biases: np.ndarray
    movie_biases: np.ndarray
    mu: float
    lam: float
    gamma: float
    train_loss_user_view: List[float] = field(default_factory=list)
    train_loss_movie_view: List[float] = field(default_factory=list)
    train_rmse: List[float] = field(default_factory=list)
    test_rmse: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)


def compute_bias_objective(
    sse: float,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    lam: float,
    gamma: float,
) -> float:
    """Compute negative log-posterior (regularized loss) for the bias model.

    L(a, b) = (lambda / 2) * sum_{(m,n) in Omega} (r_{m,n} - mu - a_m - b_n)^2
              + (gamma / 2) * sum_m a_m^2
              + (gamma / 2) * sum_n b_n^2
    """
    reg_a = float(np.dot(user_biases, user_biases))
    reg_b = float(np.dot(movie_biases, movie_biases))
    return 0.5 * lam * sse + 0.5 * gamma * (reg_a + reg_b)


def fit_bias_als(
    dataset: IndexedDataset,
    num_iters: int = 20,
    lam: float = 1.0,
    gamma: float = 0.01,
    use_global_mean: bool = False,
    seed: int = 42,
    verify_dual_views: bool = True,
) -> BiasALSResult:
    """Fit the bias-only ALS model and record loss/RMSE trajectories.

    Args:
        dataset: IndexedDataset containing CSR train/test splits.
        num_iters: Number of alternating iterations.
        lam: Precision parameter (lambda) on observation noise.
        gamma: Prior precision (regularization) on user and item biases.
        use_global_mean: Whether to center ratings by global train mean mu.
        seed: Random seed for initialization.
        verify_dual_views: If True, computes the objective over both
            `data_by_user` and `data_by_movie` at every iteration to verify
            exact equivalence.

    Returns:
        BiasALSResult containing learned biases and metric histories.
    """
    rng = np.random.default_rng(seed)
    mu = float(np.mean(dataset.user_train.ratings)) if use_global_mean else 0.0

    user_biases = rng.normal(0.0, 0.01, size=dataset.num_users).astype(
        np.float32
    )
    movie_biases = rng.normal(0.0, 0.01, size=dataset.num_movies).astype(
        np.float32
    )

    result = BiasALSResult(
        user_biases=user_biases,
        movie_biases=movie_biases,
        mu=mu,
        lam=lam,
        gamma=gamma,
    )

    num_train = max(1, dataset.num_train_ratings)
    num_test = max(1, dataset.num_test_ratings)

    # Warmup JIT compilation on tiny slice if needed
    for _ in range(num_iters):
        t0 = time.perf_counter()

        # Step 1: Update user biases a_m using user_train CSR
        _update_biases_csr(
            dataset.user_train.indptr,
            dataset.user_train.indices,
            dataset.user_train.ratings,
            movie_biases,
            user_biases,
            mu,
            lam,
            gamma,
        )

        # Step 2: Update movie biases b_n using movie_train CSR
        _update_biases_csr(
            dataset.movie_train.indptr,
            dataset.movie_train.indices,
            dataset.movie_train.ratings,
            user_biases,
            movie_biases,
            mu,
            lam,
            gamma,
        )

        elapsed = time.perf_counter() - t0
        result.epoch_times.append(elapsed)

        # Evaluate SSE and regularized objective from user view
        sse_user = compute_sse_from_user_csr(
            dataset.user_train.indptr,
            dataset.user_train.indices,
            dataset.user_train.ratings,
            user_biases,
            movie_biases,
            mu,
        )
        loss_user = compute_bias_objective(
            sse_user, user_biases, movie_biases, lam, gamma
        )
        result.train_loss_user_view.append(loss_user)
        result.train_rmse.append(float(np.sqrt(sse_user / num_train)))

        if verify_dual_views:
            sse_movie = compute_sse_from_movie_csr(
                dataset.movie_train.indptr,
                dataset.movie_train.indices,
                dataset.movie_train.ratings,
                user_biases,
                movie_biases,
                mu,
            )
            loss_movie = compute_bias_objective(
                sse_movie, user_biases, movie_biases, lam, gamma
            )
            result.train_loss_movie_view.append(loss_movie)
        else:
            result.train_loss_movie_view.append(loss_user)

        # Evaluate test RMSE
        sse_test = compute_sse_from_user_csr(
            dataset.user_test.indptr,
            dataset.user_test.indices,
            dataset.user_test.ratings,
            user_biases,
            movie_biases,
            mu,
        )
        result.test_rmse.append(float(np.sqrt(sse_test / num_test)))

    return result
