from typing import Tuple, List

import numpy as np


def synchronize_sort(
        key: np.ndarray,
        dependent: np.ndarray,
        descending: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort two arrays synchronously using the sorted indices of the key array

    Arguments:
        key: 1d-array like of floats
            Array which sorting depends on
        dependent: 1d-array like of floats
            Array that will be arranged based on the sorted indices of `key`
        descending: bool, optional (default=False)
            Order to sort by

    Return:
        Array of the same type and shape as `key` in sorted order
        Array of the same type and shape as `dependent` based on the sorted indices of `key`
    """

    sorted_idx = np.argsort(key * (-1 if descending else 1))
    return key[sorted_idx], dependent[sorted_idx]


def get_bin_sizes(arr: np.ndarray, num_bins: int) -> List[int]:
    assert num_bins > 0, "Number of bins needs to be positive."
    score_bins = np.array_split(arr, num_bins)
    return [len(score_bins[i]) for i in np.arange(len(score_bins))]
