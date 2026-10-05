"""Dual Compressed Sparse Row (CSR) data indexing and metadata utilities.

Provides O(1) contiguous slicing by user (data_by_user) and by movie
(data_by_movie) for both training and test splits, as well as MovieLens
genre side-information extraction from movies.csv.
"""

import array as arr
import csv
from dataclasses import dataclass
import pickle as pkl
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CSRMatrix:
    """Compressed Sparse Row representation of a bipartite rating matrix.

    For entity index `idx` (either a user m in 0..M-1 or a movie n in 0..N-1),
    its interactions are stored contiguously in the slice
    `[offsets[idx] : offsets[idx + 1]]` of `indices` and `ratings`.
    """

    num_rows: int
    num_cols: int
    offsets: np.ndarray  # int32 array of shape (num_rows + 1,)
    indices: np.ndarray  # int32 array of shape (nnz,)
    ratings: np.ndarray  # float32 array of shape (nnz,) in star units [0.5, 5.0]

    @property
    def nnz(self) -> int:
        """Number of observed ratings in this sparse matrix."""
        return int(self.ratings.shape[0])

    @property
    def indptr(self) -> np.ndarray:
        """Standard CSR row-pointer alias for `offsets`."""
        return self.offsets

    @property
    def counts(self) -> np.ndarray:
        """Number of ratings per row entity."""
        return np.diff(self.offsets)

    def row_degrees(self) -> np.ndarray:
        """Return array of row degrees (number of ratings per row)."""
        return np.diff(self.offsets)

    def get_row(self, row_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (column_indices, ratings) for a single row in O(1) time."""
        start = self.offsets[row_idx]
        end = self.offsets[row_idx + 1]
        return self.indices[start:end], self.ratings[start:end]


@dataclass
class IndexedDataset:
    """Complete dual-indexed train/test dataset with ID mappings."""

    num_users: int
    num_movies: int
    data_by_user_train: CSRMatrix
    data_by_user_test: CSRMatrix
    data_by_movie_train: CSRMatrix
    data_by_movie_test: CSRMatrix
    users_index: Dict[str, int]
    users_index_rev: List[str]
    movies_index: Dict[str, int]
    movies_index_rev: List[str]

    @property
    def user_train(self) -> CSRMatrix:
        """Alias for `data_by_user_train`."""
        return self.data_by_user_train

    @property
    def user_test(self) -> CSRMatrix:
        """Alias for `data_by_user_test`."""
        return self.data_by_user_test

    @property
    def movie_train(self) -> CSRMatrix:
        """Alias for `data_by_movie_train`."""
        return self.data_by_movie_train

    @property
    def movie_test(self) -> CSRMatrix:
        """Alias for `data_by_movie_test`."""
        return self.data_by_movie_test

    @property
    def num_train_ratings(self) -> int:
        """Total number of training ratings."""
        return self.data_by_user_train.nnz

    @property
    def num_test_ratings(self) -> int:
        """Total number of test ratings."""
        return self.data_by_user_test.nnz


def parse_ratings_csv(
    filename: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    label_func: Optional[Callable[[int, int, int], int]] = None,
) -> Dict[str, Any]:
    """Parse a MovieLens ratings CSV into dual contiguous CSR-like arrays.

    Ratings are stored compactly in half-star uint8 units (2 * rating in 1..10)
    during serialization and converted to float32 star units upon loading.
    """
    rng = random.Random(seed)
    if label_func is None:
        label_func = lambda u, m, t: 1 if rng.random() < train_ratio else 0

    user_index: Dict[str, int] = {}
    movie_index: Dict[str, int] = {}

    user_ids_mapped = arr.array("I")
    movie_ids_mapped = arr.array("I")
    ratings_mapped = arr.array("B")
    labels = arr.array("B")
    timesteps = arr.array("I")

    user_counts: List[int] = []
    movie_counts: List[int] = []

    with open(filename, "r", encoding="utf-8", buffering=4 * 1024 * 1024) as f:
        f.readline()  # Skip CSV header
        for line in f:
            row = line.rstrip().split(",")
            if len(row) < 3:
                continue
            u_str, m_str, r_str = row[0], row[1], row[2]
            tmstp = int(row[3]) if len(row) > 3 else 0

            u_id = user_index.get(u_str)
            if u_id is None:
                u_id = len(user_index)
                user_index[u_str] = u_id
                user_counts.append(0)

            m_id = movie_index.get(m_str)
            if m_id is None:
                m_id = len(movie_index)
                movie_index[m_str] = m_id
                movie_counts.append(0)

            user_ids_mapped.append(u_id)
            movie_ids_mapped.append(m_id)
            timesteps.append(tmstp)
            ratings_mapped.append(int(round(2.0 * float(r_str))))
            labels.append(label_func(u_id, m_id, tmstp))

            user_counts[u_id] += 1
            movie_counts[m_id] += 1

    num_users = len(user_index)
    num_movies = len(movie_index)
    num_ratings = len(ratings_mapped)

    users_offsets = arr.array("I", [0] * (num_users + 1))
    movies_offsets = arr.array("I", [0] * (num_movies + 1))

    acc = 0
    for i in range(num_users):
        acc += user_counts[i]
        users_offsets[i + 1] = acc

    acc = 0
    for i in range(num_movies):
        acc += movie_counts[i]
        movies_offsets[i + 1] = acc

    users_movies = arr.array("I", [0] * num_ratings)
    users_ratings = arr.array("B", [0] * num_ratings)
    users_labels = arr.array("B", [0] * num_ratings)

    movies_users = arr.array("I", [0] * num_ratings)
    movies_ratings = arr.array("B", [0] * num_ratings)
    movies_labels = arr.array("B", [0] * num_ratings)

    user_positions = list(users_offsets[:-1])
    movie_positions = list(movies_offsets[:-1])

    for i in range(num_ratings):
        u = user_ids_mapped[i]
        m = movie_ids_mapped[i]
        r = ratings_mapped[i]
        lbl = labels[i]

        u_pos = user_positions[u]
        users_movies[u_pos] = m
        users_ratings[u_pos] = r
        users_labels[u_pos] = lbl
        user_positions[u] += 1

        m_pos = movie_positions[m]
        movies_users[m_pos] = u
        movies_ratings[m_pos] = r
        movies_labels[m_pos] = lbl
        movie_positions[m] += 1

    user_index_rev = [""] * num_users
    for k, v in user_index.items():
        user_index_rev[v] = k

    movie_index_rev = [""] * num_movies
    for k, v in movie_index.items():
        movie_index_rev[v] = k

    return {
        "users_index": user_index,
        "users_index_rev": user_index_rev,
        "users_ratings": users_ratings,
        "users_offsets": users_offsets,
        "users_movies": users_movies,
        "users_labels": users_labels,
        "movies_index": movie_index,
        "movies_index_rev": movie_index_rev,
        "movies_ratings": movies_ratings,
        "movies_offsets": movies_offsets,
        "movies_users": movies_users,
        "movies_labels": movies_labels,
    }


