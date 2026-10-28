# Scalable Probabilistic Matrix Factorization & Genre-Augmented ALS on MovieLens-32M

**Author:** Joel Gedeon Asa Mapa  
**Institution:** African Institute for Mathematical Sciences (AIMS) South Africa  
**Course:** Applied Machine Learning at Scale  

This repository contains a high-performance, PEP 8 compliant implementation of Alternating Least Squares (ALS) Matrix Factorization and Hybrid Genre-Augmented Matrix Factorization scaled to the full **MovieLens-32M** benchmark ($32,000,204$ ratings across $200,948$ users and $84,432$ movies).

---

## Repository Structure

```text
├── src/
│   ├── __init__.py
│   ├── data_indexing.py      # Dual CSR (user & movie) zero-copy indexing & genre metadata parser
│   ├── als_biases.py         # Practical 2: Bias-only ALS model & dual-view loss invariance checks
│   ├── als_latent.py         # Practicals 3 & 4: Numba parallel CSR Cholesky ALS, serial & batched NumPy
│   ├── als_genres.py         # Practical 5: 3-block Hybrid Genre-Augmented ALS for cold-start items
│   └── recommendations.py    # Practical 4: Dummy user closed-form inference, ranking & polarization
├── scripts/
│   ├── run_practicals_1_2.py # Generates Practical 1 & 2 figures and numerical summaries
│   ├── run_practicals_3_4a.py# Generates Practical 3 & 4A convergence, scaling, overfitting & benchmarks
│   ├── run_practical_4b.py   # Full 32M training, 2D movie map, dummy user & polarizing movies
│   └── run_practical_5.py    # Hybrid Genre-ALS training, cold-start evaluation & 2D genre map
├── report/
│   ├── main.tex              # ICML 2023 formatted technical report (8+ pages + technical appendices)
│   ├── references.bib        # BibTeX bibliography
│   ├── figures/              # Publication-quality vector PDF plots (no internal figure titles)
│   └── main.pdf              # Compiled technical report
└── report.pdf                # Root copy of compiled ICML 2023 technical report
```

## Key Technical Highlights

1. **Dual CSR Sparse Indexing (`src/data_indexing.py`)**: Replaces nested Python lists with contiguous `int32`/`float32` Compressed Sparse Row arrays (`indptr`, `indices`, `ratings`), reducing RAM usage for 32M ratings to `< 520 MB` and loading in `< 5` seconds.
2. **Lock-Free Parallel Cholesky ALS (`src/als_latent.py`)**: Uses `@numba.njit(parallel=True, fastmath=True)` over CSR rows with thread-local normal equation accumulation and an in-place $L L^\top$ Cholesky solver, avoiding heap allocations and LAPACK call overhead for small $K \le 32$.
3. **Hybrid Genre-Informed Priors (`src/als_genres.py`)**: Centers each movie's Gaussian prior at $\bar{v}_n = |F(n)|^{-1/2} \sum_{g \in F(n)} f_g$, enabling non-trivial latent representations and accurate predictions even for cold-start movies with zero training ratings ($|\Omega_n| = 0$).
