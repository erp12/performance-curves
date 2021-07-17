"""Utilities that support the functionality of the `performance_curve` package."""
from typing import Tuple, List, Callable, Collection, Mapping

import numpy as np
from pyrsistent import PClass, pmap_field


def all_same_size(collections: List[Collection]) -> bool:
    """Return True if all collections have the same size/length and False otherwise.

    Args:
        collections: A list of collections.

    Returns:
        Indication if all collections are the same size.
    """
    sizes = [len(c) for c in collections]
    return all([size == sizes[0] for size in sizes])


class SynchronizedArrays(PClass):
    """A collection of arrays that must stay synchronized.

    If arrays are "synchronized", that means that the elements at the same position
    in each array are associated. If order, membership, or any other transformation
    is applied to one array, a corresponding transformation must be applied to the
    other arrays such that

    Synchronized arrays must all be the same length.

    Args:
        arrays (Mapping[object, np.ndarray]): A mapping of array IDs to 1-dimensional arrays.

    """

    arrays: Mapping[object, np.ndarray] = pmap_field(
        key_type=object,
        value_type=np.ndarray,
        invariant=lambda m: (
            all_same_size(m.values()),
            "All synchronized arrays must be the same length.",
        ),
    )

    def sort(self, array_key: object, descending: bool = False) -> "SynchronizedArrays":
        """Sort the arrays based on the natural sorting of one of the arrays.

        Args:
            array_key: The array to drive the sorting.
            descending: If True, sort by descending, else ascending.

        Returns:
            A new `SynchronizedArrays` object with the arrays sorted.

        """
        sorted_idx = np.argsort(self.arrays[array_key] * (-1 if descending else 1))
        return SynchronizedArrays(arrays={k: v[sorted_idx] for k, v in self.arrays.items()})

    def filter(self, array_id: object, pred: Callable) -> "SynchronizedArrays":
        """Filter the arrays based on a predicate applied to one of the arrays.

        Args:
            array_id: The array to drive the filtering.
            pred: The predicate to apply to the key array.

        Returns:
            A new `SynchronizedArrays` object with the arrays filtered.

        """
        filtered_idx = np.where(pred(self.arrays[array_id]))
        return SynchronizedArrays(arrays={k: v[filtered_idx] for k, v in self.arrays.items()})

    def __getitem__(self, k: object) -> np.ndarray:
        """Get the array under the corresponding identifier/key.

        Args:
            k: The identifier, or key, of the array.

        Returns:
            The array.

        """
        return self.arrays[k]


def synchronize_sort(key: np.ndarray, dependent: np.ndarray, descending: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Sort two arrays synchronously using the sorted indices of the `key` array to sort the `dependent` array.

    Note: This function assumes 1-dimensional arrays.

    Arguments:
        key: Array which drives the sorting of both itself and the `dependent` array.
        dependent: Array that will be arranged based on the sorted indices of `key`.
        descending: If `True`, sorts in descending order. Otherwise sorts in ascending order.

    Returns:
        A tuple containing two arrays. The first is the `key` array in sorted order and the second
        is `dependent` based on the sorted indices of `key`.
    """
    sorted_idx = np.argsort(key * (-1 if descending else 1))
    return key[sorted_idx], dependent[sorted_idx]


def get_bin_sizes(arr: np.ndarray, num_bins: int) -> List[int]:
    """A list describing the size of each bin if the array is split into `n` bins.

    Array splitting is a best-effort at creating equal sized bins, however if the size of the array
    is not a multiple of `num_bins`, some of the bins will be one element smaller.

    Args:
        arr: The array being split.
        num_bins: The number of bins to split into.

    Returns:
        Sequence of bin sizes.
    """
    assert num_bins > 0, "Number of bins needs to be positive."
    score_bins = np.array_split(arr, num_bins)
    return [len(score_bins[i]) for i in np.arange(len(score_bins))]
