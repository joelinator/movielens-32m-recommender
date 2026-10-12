# MovieLens-32M Collaborative Filtering Practicals

Implementation of sparse CSR data indexing, Bias-Only ALS, and Numba-accelerated Matrix Factorization on MovieLens-5M and MovieLens-32M.

## Structure
- `src/data_indexing.py`: Dual CSR user/movie indexing and train/test splitting.
- `src/als_biases.py`: Bias-only Alternating Least Squares model.
- `src/als_latent.py`: Matrix Factorization (PMF) with batched NumPy and Numba Cholesky solvers.
