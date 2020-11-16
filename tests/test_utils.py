import numpy as np
from performance_curves.utils import synchronize_sort, get_bin_sizes


def test_synchronize_sort():
    # Given a key and a dependent array of equal length
    key = np.array([0.1, 0.4, 0.91, 0.8, 0.05])
    dependent = np.array([0, 0, 1, 1, 0])
    sorted_key, sorted_dependent = synchronize_sort(key, dependent, descending=True)
    np.testing.assert_array_equal(sorted_key, np.array([0.91, 0.8, 0.4, 0.1, 0.05]))
    np.testing.assert_array_equal(sorted_dependent, np.array([1, 1, 0, 0, 0]))

    # Given a key and a dependent array where the former is longer than the latter
    key = np.array([0.1, 0.4, 0.91])
    dependent = np.array([0, 0, 1, 1, 0])
    sorted_key, sorted_dependent = synchronize_sort(key, dependent, descending=True)
    np.testing.assert_array_equal(sorted_key, np.array([0.91, 0.4, 0.1]))
    np.testing.assert_array_equal(sorted_dependent, np.array([1, 0, 0]))

    # Given a key and a dependent array where the former is shorter than the latter
    key = np.array([0.1, 0.4, 0.91, 0.8, 0.05])
    dependent = np.array([0, 1])
    with np.testing.assert_raises(IndexError):
        synchronize_sort(key, dependent, descending=True)

    # Given an empty key array and a non-empty dependent array
    key = np.array([])
    dependent = np.array([0, 0, 1, 1, 0])
    sorted_key, sorted_dependent = synchronize_sort(key, dependent, descending=True)
    assert len(sorted_key) == 0
    assert len(sorted_dependent) == 0

    # Given a non-empty key array and an empty dependent array
    key = np.array([0.1, 0.4, 0.91])
    dependent = np.array([])
    with np.testing.assert_raises(IndexError):
        synchronize_sort(key, dependent, descending=True)

    # Given two empty arrays
    key = np.array([])
    dependent = np.array([])
    sorted_key, sorted_dependent = synchronize_sort(key, dependent, descending=True)
    assert len(sorted_key) == 0
    assert len(sorted_dependent) == 0


def test_get_bin_sizes():
    # Given a normal score array and the number of bins for it to be split into is smaller than array length
    scores = np.array([0.1, 0.4, 0.91, 0.8, 0.05])
    bin_sizes = get_bin_sizes(scores, 3)
    np.testing.assert_array_equal(bin_sizes, [2, 2, 1])

    # Given an empty score array to be split
    scores = np.array([])
    bin_sizes = get_bin_sizes(scores, 3)
    np.testing.assert_array_equal(bin_sizes, [0, 0, 0])

    # Given a normal score array to be split into 0 bins
    scores = np.array([0.1, 0.4, 0.91, 0.8, 0.05])
    with np.testing.assert_raises(AssertionError):
        get_bin_sizes(scores, 0)

    # Given a normal score array and the number of bins for it to be split into is larger than array length
    scores = np.array([0.1, 0.4, 0.91])
    bin_sizes = get_bin_sizes(scores, 5)
    np.testing.assert_array_equal(bin_sizes, [1, 1, 1, 0, 0])
