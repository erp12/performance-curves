import numpy as np
from performance_curves.utils import synchronize_sort, get_bin_sizes


def test_synchronize_sort():
    key = np.array([0.1, 0.4, 0.91, 0.8, 0.05])
    dependent = np.array([0, 0, 1, 1, 0])

    sorted_key, sorted_dependent = synchronize_sort(key, dependent, descending=True)

    np.testing.assert_array_equal(sorted_key, np.array([0.91, 0.8, 0.4, 0.1, 0.05]))
    np.testing.assert_array_equal(sorted_dependent, np.array([1, 1, 0, 0, 0]))


def test_get_bin_sizes():
    scores = np.array([0.1, 0.4, 0.91, 0.8, 0.05])
    bin_sizes = get_bin_sizes(scores, 3)
    np.testing.assert_array_equal(bin_sizes, [2, 2, 1])
