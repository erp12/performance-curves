import numpy as np

from performance_curves.performance_curve_bundle import PerformanceCurveBundle
from performance_curves.metric import RECALL


def test_performance_curve_bundle():
    y_true = np.array([0, 1, 0, 0, 0, 1])
    method_scores = {"model_1": np.array([0.05, 0.19, 0.02, 0.16, 0.26, 0.60]),
                     "model_2": np.array([0.08, 0.27, 0.05, 0.27, 0.23, 0.22]),
                     "model_3": np.array([0.01, 0.89, 0.02, 0.04, 0.76, 0.99])}

    # A bundle that includes 3 normal performance curves, a random and a perfect curve for comparison with no binning
    np.random.seed(42)
    raw_bundle = PerformanceCurveBundle(y_true, method_scores, RECALL, num_trials=3)
    curves = raw_bundle.curves
    curve_1 = curves["model_1"]
    curve_2 = curves["model_2"]
    curve_3 = curves["model_3"]
    random_curve = raw_bundle.random_curve
    perfect_curve = raw_bundle.perfect_curve

    case_counts = np.arange(1, 7)
    np.testing.assert_array_equal(curve_1.performance_values, np.array([0.5, 0.5, 1., 1., 1., 1.]))
    np.testing.assert_array_equal(curve_1.case_counts, case_counts)

    np.testing.assert_array_equal(curve_2.performance_values, np.array([0.5, 0.5, 0.5, 1., 1., 1.]))
    np.testing.assert_array_equal(curve_2.case_counts, case_counts)

    np.testing.assert_array_equal(curve_3.performance_values, np.array([0.5, 1., 1., 1., 1., 1.]))
    np.testing.assert_array_equal(curve_3.case_counts, case_counts)

    np.testing.assert_array_almost_equal(random_curve.performance_values, np.array([0., 0.166, 0.5, 0.833, 0.833, 1.]), decimal=3)
    np.testing.assert_array_equal(random_curve.case_counts, case_counts)

    np.testing.assert_array_equal(perfect_curve.performance_values, np.array([0.5, 1., 1., 1., 1., 1.]))
    np.testing.assert_array_equal(perfect_curve.case_counts, case_counts)

    # A bundle without a random performance curve
    raw_bundle_wo_random = PerformanceCurveBundle(y_true, method_scores, RECALL, make_random=False)
    assert raw_bundle_wo_random.random_curve is None

    # A bundle without a perfect performance curve
    raw_bundle_wo_perfect = PerformanceCurveBundle(y_true, method_scores, RECALL, make_perfect=False)
    assert raw_bundle_wo_perfect.perfect_curve is None

    # A bundle with binning is turned on
    np.random.seed(42)
    binned_bundle = PerformanceCurveBundle(y_true, method_scores, RECALL, num_bins=2, num_trials=3)
    binned_curves = binned_bundle.curves
    binned_1 = binned_curves["model_1"]
    binned_2 = binned_curves["model_2"]
    binned_3 = binned_curves["model_3"]
    binned_random_curve = binned_bundle.random_curve
    binned_perfect_curve = binned_bundle.perfect_curve

    case_counts = [3, 6]
    np.testing.assert_array_equal(binned_1.performance_values, np.array([1., 1.]))
    np.testing.assert_array_equal(binned_1.case_counts, case_counts)

    np.testing.assert_array_equal(binned_2.performance_values, np.array([0.5, 1.]))
    np.testing.assert_array_equal(binned_2.case_counts, case_counts)

    np.testing.assert_array_equal(binned_3.performance_values, np.array([1., 1.]))
    np.testing.assert_array_equal(binned_3.case_counts, case_counts)

    np.testing.assert_array_equal(binned_random_curve.performance_values, np.array([0.5, 1.]))
    np.testing.assert_array_equal(binned_random_curve.case_counts, case_counts)

    np.testing.assert_array_equal(binned_perfect_curve.performance_values, np.array([1., 1.]))
    np.testing.assert_array_equal(binned_perfect_curve.case_counts, case_counts)
