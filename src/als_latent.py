"""Matrix Factorization with Biases and Latent Factors (Practical 3).

Implements Probabilistic Matrix Factorization (PMF) trained via Alternating
Least Squares (ALS) using vectorized batched NumPy solves:
    r_hat_{m,n} = mu + a_m + b_n + u_m^T v_n
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List

import numpy as np

from src.data_indexing import IndexedDataset


def update_biases_and_factors_numpy_batched(
    indptr: np.ndarray,
    indices: np.ndarray,
    ratings: np.ndarray,
    target_biases: np.ndarray,
    other_biases: np.ndarray,
    target_factors: np.ndarray,
    other_factors: np.ndarray,
    mu: float,
    lam: float,
    gamma: float,
    tau: float,
    batch_size: int = 512,
) -> None:
    """Vectorized batched NumPy solver (`np.add.reduceat` + `np.linalg.solve`)."""
    num_rows = len(indptr) - 1
    k_dim = target_factors.shape[1]
    eye_k = np.eye(k_dim, dtype=np.float32) * np.float32(tau)

    for b_start in range(0, num_rows, batch_size):
        b_end = min(b_start + batch_size, num_rows)
        sub_ptr = indptr[b_start : b_end + 1]
        edge_start = int(sub_ptr[0])
        edge_end = int(sub_ptr[-1])
        if edge_end == edge_start:
            target_biases[b_start:b_end] = 0.0
            target_factors[b_start:b_end] = 0.0
            continue

        local_ptr = sub_ptr - edge_start
        counts = np.diff(local_ptr).astype(np.float32)
        active_mask = counts > 0

        batch_indices = indices[edge_start:edge_end]
        batch_ratings = ratings[edge_start:edge_end]
        row_ids = np.repeat(
            np.arange(b_start, b_end, dtype=np.int32), counts.astype(np.int32)
        )

        dots = np.sum(
            target_factors[row_ids] * other_factors[batch_indices], axis=1
        )
        res_b = batch_ratings - mu - other_biases[batch_indices] - dots

        non_empty_indices = np.where(active_mask)[0]
        starts_non_empty = local_ptr[:-1][active_mask]
        sums_b = np.zeros(b_end - b_start, dtype=np.float32)
        sums_b[non_empty_indices] = np.add.reduceat(res_b, starts_non_empty)

        new_biases = np.zeros(b_end - b_start, dtype=np.float32)
        new_biases[active_mask] = (lam * sums_b[active_mask]) / (
            lam * counts[active_mask] + gamma
        )
        target_biases[b_start:b_end] = new_biases

        r_tilde = (
            batch_ratings
            - mu
            - target_biases[row_ids]
            - other_biases[batch_indices]
        )
        v_batch = other_factors[batch_indices]
        weighted_v = (lam * r_tilde)[:, None] * v_batch
        outer_v = lam * (v_batch[:, :, None] * v_batch[:, None, :])

        rhs = np.zeros((b_end - b_start, k_dim), dtype=np.float32)
        rhs[non_empty_indices] = np.add.reduceat(
            weighted_v, starts_non_empty, axis=0
        )

        lhs = np.broadcast_to(
            eye_k, (b_end - b_start, k_dim, k_dim)
        ).copy()
        lhs[non_empty_indices] += np.add.reduceat(
            outer_v, starts_non_empty, axis=0
        )

        solved = np.zeros((b_end - b_start, k_dim), dtype=np.float32)
        solved[active_mask] = np.linalg.solve(
            lhs[active_mask], rhs[active_mask][..., None]
        )[..., 0]
        target_factors[b_start:b_end] = solved


def compute_pmf_sse_numpy(
    indptr: np.ndarray,
    indices: np.ndarray,
    ratings: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    user_factors: np.ndarray,
    movie_factors: np.ndarray,
    mu: float,
) -> float:
    """Compute sum of squared errors (SSE) in chunks using NumPy."""
    counts = np.diff(indptr)
    user_ids = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    total_sse = 0.0
    chunk = 4_000_000
    for s in range(0, len(ratings), chunk):
        e = min(s + chunk, len(ratings))
        u_c = user_ids[s:e]
        m_c = indices[s:e]
        dots = np.sum(user_factors[u_c] * movie_factors[m_c], axis=1)
        err = ratings[s:e] - (mu + user_biases[u_c] + movie_biases[m_c] + dots)
        total_sse += float(np.dot(err, err))
    return total_sse


def compute_pmf_objective(
    sse: float,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    user_factors: np.ndarray,
    movie_factors: np.ndarray,
    lam: float,
    gamma: float,
    tau: float,
) -> float:
    """Compute negative log-posterior (regularized loss) for the PMF model."""
    reg_biases = float(
        np.dot(user_biases, user_biases) + np.dot(movie_biases, movie_biases)
    )
    reg_factors = float(
        np.sum(user_factors.astype(np.float64) ** 2)
        + np.sum(movie_factors.astype(np.float64) ** 2)
    )
    return 0.5 * lam * sse + 0.5 * gamma * reg_biases + 0.5 * tau * reg_factors


@dataclass
class PMFALSResult:
    """Training history and learned parameters for PMF ALS."""

    user_biases: np.ndarray
    movie_biases: np.ndarray
    user_factors: np.ndarray
    movie_factors: np.ndarray
    mu: float
    k_dim: int
    lam: float
    gamma: float
    tau: float
    train_loss: List[float] = field(default_factory=list)
    train_rmse: List[float] = field(default_factory=list)
    test_rmse: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)


def fit_pmf_als(
    dataset: IndexedDataset,
    k_dim: int = 10,
    num_iters: int = 15,
    lam: float = 1.0,
    gamma: float = 0.05,
    tau: float = 5.0,
    use_global_mean: bool = False,
    seed: int = 42,
    eval_every: int = 1,
) -> PMFALSResult:
    """Train the PMF latent factor model via Batched NumPy ALS."""
    rng = np.random.default_rng(seed)
    mu = float(np.mean(dataset.user_train.ratings)) if use_global_mean else 0.0

    user_biases = np.zeros(dataset.num_users, dtype=np.float32)
    movie_biases = np.zeros(dataset.num_movies, dtype=np.float32)

    scale = 1.0 / np.sqrt(max(1, k_dim))
    user_factors = rng.normal(
        0.0, scale * 0.1, size=(dataset.num_users, k_dim)
    ).astype(np.float32)
    movie_factors = rng.normal(
        0.0, scale * 0.1, size=(dataset.num_movies, k_dim)
    ).astype(np.float32)

    result = PMFALSResult(
        user_biases=user_biases,
        movie_biases=movie_biases,
        user_factors=user_factors,
        movie_factors=movie_factors,
        mu=mu,
        k_dim=k_dim,
        lam=lam,
        gamma=gamma,
        tau=tau,
    )

    num_train = max(1, dataset.num_train_ratings)
    num_test = max(1, dataset.num_test_ratings)

    for epoch in range(num_iters):
        t0 = time.perf_counter()
        update_biases_and_factors_numpy_batched(
            dataset.user_train.indptr,
            dataset.user_train.indices,
            dataset.user_train.ratings,
            user_biases,
            movie_biases,
            user_factors,
            movie_factors,
            mu,
            lam,
            gamma,
            tau,
        )
        update_biases_and_factors_numpy_batched(
            dataset.movie_train.indptr,
            dataset.movie_train.indices,
            dataset.movie_train.ratings,
            movie_biases,
            user_biases,
            movie_factors,
            user_factors,
            mu,
            lam,
            gamma,
            tau,
        )
        result.epoch_times.append(time.perf_counter() - t0)

        if (epoch + 1) % eval_every == 0 or epoch == num_iters - 1:
            sse_train = compute_pmf_sse_numpy(
                dataset.user_train.indptr,
                dataset.user_train.indices,
                dataset.user_train.ratings,
                user_biases,
                movie_biases,
                user_factors,
                movie_factors,
                mu,
            )
            loss = compute_pmf_objective(
                sse_train,
                user_biases,
                movie_biases,
                user_factors,
                movie_factors,
                lam,
                gamma,
                tau,
            )
            sse_test = compute_pmf_sse_numpy(
                dataset.user_test.indptr,
                dataset.user_test.indices,
                dataset.user_test.ratings,
                user_biases,
                movie_biases,
                user_factors,
                movie_factors,
                mu,
            )
            result.train_loss.append(loss)
            result.train_rmse.append(float(np.sqrt(sse_train / num_train)))
            result.test_rmse.append(float(np.sqrt(sse_test / num_test)))

    return result
