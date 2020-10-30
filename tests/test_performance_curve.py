import numpy as np

from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from performance_curves.performance_curve import performance_curve


def test_performance_curve():
    #
    # Givens
    #

    # two arrays of ground truths and predicted scores, correspondingly
    y_true = np.array([0, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.91, 0.8, 0.05])

    #
    # Whens
    #

    # measuring the performance curves of all data points
    x1 = performance_curve(y_true, y_score, precision_score)
    x2 = performance_curve(y_true, y_score, recall_score)
    x3 = performance_curve(y_true, y_score, accuracy_score)
    x4 = performance_curve(y_true, y_score, roc_auc_score)

    #
    # Thens
    #
    sorted_y_true = np.array([0.91, 0.8, 0.4, 0.1, 0.05])
    np.testing.assert_array_equal(x1.score_thresholds, sorted_y_true)
    np.testing.assert_array_equal(x2.score_thresholds, sorted_y_true)
    np.testing.assert_array_equal(x3.score_thresholds, sorted_y_true)
    np.testing.assert_array_equal(x4.score_thresholds, sorted_y_true)

    np.testing.assert_array_almost_equal(x1.performance_values, np.array([1., 1., 0.666, 0.5, 0.4]), decimal=3)
    np.testing.assert_array_almost_equal(x2.performance_values, np.array([0.5, 1., 1., 1., 1.]), decimal=3)
    np.testing.assert_array_almost_equal(x3.performance_values, np.array([0.8, 1, 0.8, 0.6, 0.4]), decimal=3)
    np.testing.assert_array_almost_equal(x4.performance_values, np.array([0.75, 1, 0.833, 0.666, 0.5]), decimal=3)

    num_cases = np.array([i for i in range(1, 6)])
    np.testing.assert_array_equal(x1.case_counts, num_cases)
    np.testing.assert_array_equal(x2.case_counts, num_cases)
    np.testing.assert_array_equal(x3.case_counts, num_cases)
    np.testing.assert_array_equal(x4.case_counts, num_cases)

    #
    # Whens
    #

    # dividing the data points into 3 bins and measuring their performance curves
    x1 = performance_curve(y_true, y_score, precision_score, num_bins=3)
    x2 = performance_curve(y_true, y_score, recall_score, num_bins=3)
    x3 = performance_curve(y_true, y_score, accuracy_score, num_bins=3)
    x4 = performance_curve(y_true, y_score, roc_auc_score, num_bins=3)

    #
    # Thens
    #
    sorted_y_true_ub = np.array([0.91, 0.4, 0.05])
    np.testing.assert_array_equal(x1.score_thresholds, sorted_y_true_ub)
    np.testing.assert_array_equal(x2.score_thresholds, sorted_y_true_ub)
    np.testing.assert_array_equal(x3.score_thresholds, sorted_y_true_ub)
    np.testing.assert_array_equal(x4.score_thresholds, sorted_y_true_ub)

    np.testing.assert_array_almost_equal(x1.performance_values, np.array([1., 0.5, 0.4]), decimal=3)
    np.testing.assert_array_almost_equal(x2.performance_values, np.array([1., 1., 1.]), decimal=3)
    np.testing.assert_array_almost_equal(x3.performance_values, np.array([1., 0.6, 0.4]), decimal=3)
    np.testing.assert_array_almost_equal(x4.performance_values, np.array([1., 0.666, 0.5]), decimal=3)

    num_cases = np.array([2, 4, 5])
    np.testing.assert_array_equal(x1.case_counts, num_cases)
    np.testing.assert_array_equal(x2.case_counts, num_cases)
    np.testing.assert_array_equal(x3.case_counts, num_cases)
    np.testing.assert_array_equal(x4.case_counts, num_cases)
