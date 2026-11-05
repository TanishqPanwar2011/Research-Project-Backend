"""Orthogonal arrays for 2-level factors, built from a Hadamard matrix."""
import numpy as np


def hadamard(n: int) -> np.ndarray:
    """Sylvester Hadamard matrix; n must be a power of two."""
    h = np.array([[1]])
    while h.shape[0] < n:
        h = np.kron(np.array([[1, 1], [1, -1]]), h)
    return h


def orthogonal_array(n_factors: int) -> np.ndarray:
    """Return an (N, n_factors) array of 0/1 values.

    Every column is balanced, and every pair of columns shows all four combinations
    (00, 01, 10, 11) equally often. So N runs cover the space of feature combinations
    evenly, instead of N random guesses. Here 1 means "keep this feature".
    """
    n_runs = 2
    while n_runs - 1 < n_factors:
        n_runs *= 2
    h = hadamard(n_runs)
    return (h[:, 1 : n_factors + 1] == 1).astype(int)
