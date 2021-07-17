import pytest

import numpy as np
from pyrsistent import InvariantException

# noinspection PyProtectedMember
from performance_curves._utils import synchronize_sort, get_bin_sizes, SynchronizedArrays


@pytest.fixture
def synced_arrays() -> SynchronizedArrays:
    return SynchronizedArrays(arrays={"a": np.array([1, 2, 3]), "b": np.array([5, 4, 3])})


class TestSynchronizedArrays:

    def test_invariant(self):
        with pytest.raises(InvariantException):
            SynchronizedArrays(arrays={"a": np.arange(1), "b": np.arange(2)})

    def test_sort(self, synced_arrays: SynchronizedArrays):
        arrays = synced_arrays.sort("a", descending=True).arrays
        assert np.array_equiv(arrays["a"], np.array([3, 2, 1]))
        assert np.array_equiv(arrays["b"], np.array([3, 4, 5]))

        arrays = SynchronizedArrays(arrays={"a": np.array([]), "b": np.array([])}).sort("b").arrays
        assert np.array_equiv(arrays["a"], np.array([]))
        assert np.array_equiv(arrays["b"], np.array([]))

    def test_filter(self, synced_arrays: SynchronizedArrays):
        arrays = synced_arrays.filter("b", lambda x: x == 5).arrays
        assert np.array_equiv(arrays["a"], np.array([1]))
        assert np.array_equiv(arrays["b"], np.array([5]))

        arrays = SynchronizedArrays(arrays={"a": np.array([]), "b": np.array([])}).filter("b", lambda x: x == 1).arrays
        assert np.array_equiv(arrays["a"], np.array([]))
        assert np.array_equiv(arrays["b"], np.array([]))


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


def test_bin_sizes():
    # Given a normal score array and the number of bins for it to be split into is smaller than array length
    scores = np.array([0.1, 0.4, 0.91, 0.8, 0.05])
    sizes = get_bin_sizes(scores, 3)
    np.testing.assert_array_equal(sizes, [2, 2, 1])

    # Given an empty score array to be split
    scores = np.array([])
    sizes = get_bin_sizes(scores, 3)
    np.testing.assert_array_equal(sizes, [0, 0, 0])

    # Given a normal score array to be split into 0 bins
    scores = np.array([0.1, 0.4, 0.91, 0.8, 0.05])
    with np.testing.assert_raises(AssertionError):
        get_bin_sizes(scores, 0)

    # Given a normal score array and the number of bins for it to be split into is larger than array length
    scores = np.array([0.1, 0.4, 0.91])
    sizes = get_bin_sizes(scores, 5)
    np.testing.assert_array_equal(sizes, [1, 1, 1, 0, 0])
