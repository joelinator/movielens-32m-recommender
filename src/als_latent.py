"""Matrix Factorization with Biases and Latent Factors (Practicals 3 & 4).

Implements Probabilistic Matrix Factorization (PMF) trained via Alternating
Least Squares (ALS):
    r_hat_{m,n} = mu + a_m + b_n + u_m^T v_n

Provides three execution backends for benchmarking (Practical 4 bonus):
1. `numba_parallel`: Multi-threaded CSR row-parallel Cholesky solver (`prange`)
   with O(K^2) thread-local buffers and zero heap allocation overhead.
2. `numba_serial`: Single-threaded CSR Cholesky solver for parallel speedup
   comparison.
3. `numpy_batched`: Vectorized batched NumPy solver (`np.linalg.solve`) over
   CSR slices for comparing Python/NumPy memory bandwidth vs Numba JIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple

import numba
import numpy as np

from src.data_indexing import IndexedDataset


@numba.njit(fastmath=True)
def _solve_spd_cholesky_inplace(a_mat: np.ndarray, b_vec: np.ndarray) -> None:
    """Solve A x = b in-place for symmetric positive-definite A (K x K).

    Overwrites lower triangle of `a_mat` with Cholesky factor L (A = L L^T)
    and overwrites `b_vec` with the solution x. Avoids LAPACK call overhead
    inside tight parallel loops for small K (2 <= K <= 32).
    """
    k_dim = b_vec.shape[0]
    for i in range(k_dim):
        for j in range(i + 1):
            s = a_mat[i, j]
            for p in range(j):
                s -= a_mat[i, p] * a_mat[j, p]
            if i == j:
                # Guard against numerical underflow in float32/float64
                if s <= 1e-12:
                    s = 1e-12
                a_mat[i, j] = np.sqrt(s)
            else:
                a_mat[i, j] = s / a_mat[j, j]

    # Forward substitution: L y = b
    for i in range(k_dim):
        s = b_vec[i]
        for p in range(i):
            s -= a_mat[i, p] * b_vec[p]
        b_vec[i] = s / a_mat[i, i]

    # Backward substitution: L^T x = y
    for i in range(k_dim - 1, -1, -1):
        s = b_vec[i]
        for p in range(i + 1, k_dim):
            s -= a_mat[p, i] * b_vec[p]
        b_vec[i] = s / a_mat[i, i]



def update_biases_and_factors_serial(
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
) -> None:
    """Single-threaded CSR ALS update for benchmarking parallel speedup."""
    num_rows = len(indptr) - 1
    k_dim = target_factors.shape[1]

    a_mat = np.zeros((k_dim, k_dim), dtype=np.float64)
    b_vec = np.zeros(k_dim, dtype=np.float64)

    for i in range(num_rows):
        start = indptr[i]
        end = indptr[i + 1]
        count = end - start
        if count == 0:
            target_biases[i] = 0.0
            for d in range(k_dim):
                target_factors[i, d] = 0.0
            continue

        res_bias = 0.0
        u_i = target_factors[i]
        for idx in range(start, end):
            j = indices[idx]
            v_j = other_factors[j]
            dot_uv = 0.0
            for d in range(k_dim):
                dot_uv += u_i[d] * v_j[d]
            res_bias += ratings[idx] - mu - other_biases[j] - dot_uv

        new_bias = (lam * res_bias) / (lam * count + gamma)
        target_biases[i] = new_bias

        for d1 in range(k_dim):
            b_vec[d1] = 0.0
            for d2 in range(d1 + 1):
                a_mat[d1, d2] = 0.0
            a_mat[d1, d1] = tau

        for idx in range(start, end):
            j = indices[idx]
            v_j = other_factors[j]
            r_tilde = ratings[idx] - mu - new_bias - other_biases[j]
            lam_r = lam * r_tilde
            for d1 in range(k_dim):
                v_d1 = float(v_j[d1])
                b_vec[d1] += lam_r * v_d1
                lam_v_d1 = lam * v_d1
                for d2 in range(d1 + 1):
                    a_mat[d1, d2] += lam_v_d1 * float(v_j[d2])

        _solve_spd_cholesky_inplace(a_mat, b_vec)
        for d in range(k_dim):
            target_factors[i, d] = np.float32(b_vec[d])


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
    """Vectorized batched NumPy solver (`np.add.reduceat` + `np.linalg.solve`).

    Used to benchmark pure vectorized NumPy against Numba parallel CSR kernels.
    """
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

        # Bias update
        dots = np.sum(
            target_factors[row_ids] * other_factors[batch_indices], axis=1
        )
        res_b = batch_ratings - mu - other_biases[batch_indices] - dots

        # Guard reduceat for trailing empty rows
        non_empty_indices = np.where(active_mask)[0]
        starts_non_empty = local_ptr[:-1][active_mask]
        sums_b = np.zeros(b_end - b_start, dtype=np.float32)
        sums_b[non_empty_indices] = np.add.reduceat(res_b, starts_non_empty)

        new_biases = np.zeros(b_end - b_start, dtype=np.float32)
        new_biases[active_mask] = (lam * sums_b[active_mask]) / (
            lam * counts[active_mask] + gamma
        )
        target_biases[b_start:b_end] = new_biases

        # Factor update
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


@numba.njit(parallel=True, fastmath=True)
def compute_pmf_sse_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    ratings: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    user_factors: np.ndarray,
    movie_factors: np.ndarray,
    mu: float,
) -> float:
    """Compute sum of squared errors (SSE) for PMF model over user CSR."""
    num_users = len(indptr) - 1
    k_dim = user_factors.shape[1]
    total_sse = 0.0

    for m in numba.prange(num_users):
        start = indptr[m]
        end = indptr[m + 1]
        a_m = user_biases[m]
        u_m = user_factors[m]
        row_sse = 0.0
        for idx in range(start, end):
            n = indices[idx]
            v_n = movie_factors[n]
            dot_uv = 0.0
            for d in range(k_dim):
                dot_uv += u_m[d] * v_n[d]
            err = ratings[idx] - (mu + a_m + movie_biases[n] + dot_uv)
            row_sse += err * err
        total_sse += row_sse
    return total_sse



    sse: float,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    user_factors: np.ndarray,
    movie_factors: np.ndarray,
    lam: float,
    gamma: float,
    tau: float,
) -> float:
    """Compute negative log-posterior (regularized loss) for the PMF model.

    L(a, b, U, V) = (lambda / 2) * SSE
                    + (gamma / 2) * (||a||_2^2 + ||b||_2^2)
                    + (tau / 2) * (||U||_F^2 + ||V||_F^2)
    """
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
    backend: str = "numba_parallel",
    eval_every: int = 1,
) -> PMFALSResult:
    """Train the PMF latent factor model via Alternating Least Squares.

    Args:
        dataset: IndexedDataset with user and movie CSR splits.
        k_dim: Number of latent dimensions K.
        num_iters: Number of ALS alternating epochs.
        lam: Observation noise precision (lambda).
        gamma: Regularization precision on user/movie biases.
        tau: Regularization precision on user/movie latent vectors.
        use_global_mean: Whether to center ratings by global train mean.
        seed: Random seed for factor initialization.
        backend: One of 'numba_parallel', 'numba_serial', 'numpy_batched'.
        eval_every: Compute train/test RMSE every `eval_every` epochs.

    Returns:
        PMFALSResult with learned embeddings and training curves.
    """
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

    if backend == "numba_parallel":
    if backend == "numba_serial":
        update_fn = update_biases_and_factors_serial
    elif backend == "numpy_batched":
        update_fn = update_biases_and_factors_numpy_batched
    else:
        raise ValueError(f"Unknown backend: {backend}")

        t0 = time.perf_counter()

        # Step 1: Update user biases and user factors
        update_fn(
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

        # Step 2: Update movie biases and movie factors
        update_fn(
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

        elapsed = time.perf_counter() - t0
        result.epoch_times.append(elapsed)

        if (epoch + 1) % eval_every == 0 or epoch == num_iters - 1:
            sse_train = compute_pmf_sse_numba(
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
            sse_test = compute_pmf_sse_numba(
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
