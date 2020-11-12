import numpy as np

from performance_curves.performance_curve import PerformanceCurve, PerfectPerformanceCurve, \
    RandomPerformanceCurve, _generate_performance_values
from performance_curves.metric import RECALL, PRECISION, ACCURACY, F1


def test_generate_performance_values():
    #
    # Givens
    #
    y1 = np.array([1, 1, 0, 0, 1])
    y2 = np.array([1, 1, 1, 0, 0])

    #
    # Whens
    #
    r1 = _generate_performance_values(y1, RECALL)
    r2 = _generate_performance_values(y1, PRECISION)
    r3 = _generate_performance_values(y2, RECALL)
    r4 = _generate_performance_values(y2, PRECISION)
    r5 = _generate_performance_values(y1, ACCURACY, [3, 2])
    r6 = _generate_performance_values(y2, F1, [2, 2, 1])

    #
    # Thens
    #
    case_counts = np.arange(1, 6)
    np.testing.assert_array_almost_equal(r1[0], np.array([0.333, 0.666, 0.666, 0.666, 1.]), decimal=3)
    np.testing.assert_array_equal(r1[1], case_counts)

    np.testing.assert_array_almost_equal(r2[0], np.array([1., 1., 0.666, 0.5, 0.6]), decimal=3)
    np.testing.assert_array_equal(r2[1], case_counts)

    np.testing.assert_array_almost_equal(r3[0], np.array([0.333, 0.666, 1., 1., 1.]), decimal=3)
    np.testing.assert_array_almost_equal(r3[1], case_counts)

    np.testing.assert_array_equal(r4[0], np.array([1., 1., 1., 0.75, 0.6]))
    np.testing.assert_array_equal(r4[1], case_counts)

    np.testing.assert_array_equal(r5[0], np.array([0.6, 0.6]))
    np.testing.assert_array_equal(r5[1], np.array([3, 5]))

    np.testing.assert_array_almost_equal(r6[0], np.array([0.8, 0.857, 0.75]), decimal=3)
    np.testing.assert_array_equal(r6[1], np.array([2, 4, 5]))


def test_performance_curve():
    #
    # Givens
    #
    y_true = np.array([1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.91, 0.8, 0.05])
    y_point = np.array([3, 4, 1, 2, 5])

    #
    # Whens
    #
    x1 = PerformanceCurve(y_true, y_score, PRECISION)
    x2 = PerformanceCurve(y_true, y_score, RECALL)
    x3 = PerformanceCurve(y_true, y_score, PRECISION, num_bins=3)
    x4 = PerformanceCurve(y_true, y_point, RECALL, num_bins=2, order_descending=False)

    #
    # Thens
    #
    case_counts = np.arange(1, 6)
    np.testing.assert_array_almost_equal(x1.performance_values, np.array([1., 1., 0.666, 0.75, 0.6]), decimal=3)
    np.testing.assert_array_equal(x1.case_counts, case_counts)

    np.testing.assert_array_almost_equal(x2.performance_values, np.array([0.333, 0.666, 0.666, 1., 1.]), decimal=3)
    np.testing.assert_array_equal(x2.case_counts, case_counts)

    np.testing.assert_array_equal(x3.performance_values, np.array([1., 0.75, 0.6]))
    np.testing.assert_array_equal(x3.case_counts, np.array([2, 4, 5]))

    np.testing.assert_array_equal(x4.performance_values, np.array([1., 1.]))
    np.testing.assert_array_equal(x4.case_counts, np.array([3, 5]))


def test_perfect_performance_curve():
    #
    # Givens
    #
    y_true = np.array([0, 0, 1, 1, 0])

    #
    # Whens
    #
    x1 = PerfectPerformanceCurve(y_true, PRECISION)
    x2 = PerfectPerformanceCurve(y_true, RECALL)
    x3 = PerfectPerformanceCurve(y_true, ACCURACY, num_bins=2)

    #
    # Thens
    #
    np.testing.assert_array_almost_equal(x1.performance_values, np.array([1., 1., 0.666, 0.5, 0.4]), decimal=3)
    np.testing.assert_array_equal(x1.case_counts, np.arange(1, 6))

    np.testing.assert_array_equal(x2.performance_values, np.array([0.5, 1., 1., 1., 1.]))
    np.testing.assert_array_equal(x2.case_counts, np.arange(1, 6))

    np.testing.assert_array_equal(x3.performance_values, np.array([0.8, 0.4]))
    np.testing.assert_array_equal(x3.case_counts, np.array([3, 5]))


def test_random_performance_curve():
    #
    # Givens
    #
    y_true = np.array([0, 0, 1, 1, 0])

    #
    # Whens
    #
    x = RandomPerformanceCurve(y_true, RECALL, num_trials=3, random_seed=1)

    #
    # Thens
    #
    np.testing.assert_array_almost_equal(x.performance_values, np.array([0.5, 0.5, 0.666, 0.666, 1.]), decimal=3)
    np.testing.assert_array_equal(x.case_counts, np.arange(1, 6))
