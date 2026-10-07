"""Bias-only Alternating Least Squares (ALS) model for Practical 2.

Models predicted ratings as r_hat_{m,n} = mu + a_m + b_n.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List

import numba
import numpy as np

from src.data_indexing import IndexedDataset


@numba.njit(parallel=True, fastmath=True)
def update_user_biases_csr(
    user_indptr: np.ndarray,
    user_indices: np.ndarray,
    user_ratings: np.ndarray,
    movie_biases: np.ndarray,
    user_biases: np.ndarray,
    mu: float,
    lam: float,
    gamma: float,
) -> None:
    """Update user biases a_m in parallel over user CSR rows."""
    num_users = len(user_indptr) - 1
    for m in numba.prange(num_users):
        start = user_indptr[m]
        end = user_indptr[m + 1]
        count = end - start
        if count == 0:
            user_biases[m] = 0.0
            continue
        residual_sum = 0.0
        for idx in range(start, end):
            n = user_indices[idx]
            residual_sum += user_ratings[idx] - mu - movie_biases[n]
        user_biases[m] = (lam * residual_sum) / (lam * count + gamma)


@numba.njit(parallel=True, fastmath=True)
def update_movie_biases_csr(
    movie_indptr: np.ndarray,
    movie_indices: np.ndarray,
    movie_ratings: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    mu: float,
    lam: float,
    gamma: float,
) -> None:
    """Update movie biases b_n in parallel over movie CSR rows."""
    num_movies = len(movie_indptr) - 1
    for n in numba.prange(num_movies):
        start = movie_indptr[n]
        end = movie_indptr[n + 1]
        count = end - start
        if count == 0:
            movie_biases[n] = 0.0
            continue
        residual_sum = 0.0
        for idx in range(start, end):
            m = movie_indices[idx]
            residual_sum += movie_ratings[idx] - mu - user_biases[m]
        movie_biases[n] = (lam * residual_sum) / (lam * count + gamma)


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


@dataclass
class BiasALSResult:
    """Training history and learned parameters for Bias-Only ALS."""

    user_biases: np.ndarray
    movie_biases: np.ndarray
    mu: float
    lam: float
    gamma: float
    train_loss_user_view: List[float] = field(default_factory=list)
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
    """Compute regularized loss L(a, b) for the bias-only model."""
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
) -> BiasALSResult:
    """Fit the bias-only ALS model using separate user/movie CSR updates."""
    rng = np.random.default_rng(seed)
    mu = float(np.mean(dataset.user_train.ratings)) if use_global_mean else 0.0

    user_biases = rng.normal(0.0, 0.01, size=dataset.num_users).astype(np.float32)
    movie_biases = rng.normal(0.0, 0.01, size=dataset.num_movies).astype(np.float32)

    result = BiasALSResult(
        user_biases=user_biases,
        movie_biases=movie_biases,
        mu=mu,
        lam=lam,
        gamma=gamma,
    )

    num_train = max(1, dataset.num_train_ratings)
    num_test = max(1, dataset.num_test_ratings)

    for _ in range(num_iters):
        t0 = time.perf_counter()
        update_user_biases_csr(
            dataset.user_train.indptr,
            dataset.user_train.indices,
            dataset.user_train.ratings,
            movie_biases,
            user_biases,
            mu,
            lam,
            gamma,
        )
        update_movie_biases_csr(
            dataset.movie_train.indptr,
            dataset.movie_train.indices,
            dataset.movie_train.ratings,
            user_biases,
            movie_biases,
            mu,
            lam,
            gamma,
        )
        result.epoch_times.append(time.perf_counter() - t0)

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
