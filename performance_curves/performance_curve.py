import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from performance_curves.utils import synchronize_sort
from performance_curves.metric import Metric


class PerformanceCurveLike:
    """Stores values of the x- and y-axis of the performance curve and allows plotting"""
    def __init__(self,
                 performance_values: np.ndarray,
                 case_counts: np.ndarray,
                 metric: Metric):
        self.performance_values = performance_values
        self.case_counts = case_counts
        self.metric = metric

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.case_counts, self.performance_values)
        ax.set_xlabel('Number of Cases')
        ax.set_ylabel(self.metric.name)
        ax.set_title('Performance Curve')
        plt.show()


class PerformanceCurve(PerformanceCurveLike):
    def __init__(self,
                 performance_values: np.ndarray,
                 case_counts: np.ndarray,
                 score_thresholds: np.ndarray,
                 metric: Metric):
        super().__init__(performance_values, case_counts, metric)
        self.score_thresholds = score_thresholds

        assert len(performance_values) == len(case_counts), \
            'Lengths of performance values array and counts array do not match.'
        assert len(performance_values) == len(score_thresholds), \
            'Lengths of performance values array and scores array do not match.'

    def plot_with_thresholds(self):
        fig, ax = plt.subplots()
        ax.plot(self.score_thresholds, self.performance_values)
        ax.set_xlabel('Predicted Score (Probability Estimate)')
        ax.set_ylabel(self.metric)
        ax.set_title('Performance Curve')
        plt.show()

    def threshold_at(self, performance_point: float):
        pass


# TODO: should this function stay in this script or the utils script?
def _generate_performance_values(
        y_true: np.ndarray,
        rearranged_y_true: np.ndarray,
        metric: Metric,
        num_bins: Optional[int] = None,
        binned_array: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:

    size = len(y_true)
    result_size = num_bins if num_bins else size
    y_pred = np.zeros(size)
    results = list()
    counts = list()
    current_count = 0

    for i in range(result_size):
        if binned_array:
            num_elements = len(binned_array[i])
            first_zero = (y_pred == 0).argmax()
            y_pred[first_zero:(first_zero + num_elements)] = 1
            current_count += num_elements
        else:
            y_pred[i] = 1
            current_count += 1
        results.append(metric.func(rearranged_y_true, y_pred))
        counts.append(current_count)

    return np.array(results), np.array(counts)


def performance_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Metric,
    num_bins: Optional[int] = None
) -> PerformanceCurveLike:
    """
    Computes performance value of a metric of interest at each predicted score / probability estimate (or each upper
    bound of a range of predicted scores in case of binning) given that they are sorted in descending order

    Arguments:
        y_true: 1d-array like
            Ground truth (correct) labels.
        y_score: 1d array-like
            Predicted scores (probability estimates) as returned by a classifier.
        metric: custom metric class
            Metric class that represents a metric used to assess a classifier's prediction error for specific purposes
            given ground truth and prediction. This class's attributes include the metric name, its callable function
            (e.g. sklearn.metrics.recall_score), and whether we should minimize or maximize it to obtain the best model
            performance.
        num_bins: int, optional (default=None)
            Number of bins (or bucket) to divide the range of values into.

    Return:
        results: array of float, shape = [n_samples] or [n_bins]
            Performance values of metrics of interest when setting the score threshold at the corresponding y_score
        counts: array of int, shape = [n_samples] or [n_bins]
            Number of cases corresponding with the same position's y_score or performance_value
        sorted_y_score: array of float, shape = [n_samples] or [n_bins]
            In case num_bins is `None`, return predicted scores of all data points sorted in descending order.
            In case num_bins is not `None`, return upper bounds of the bins sorted in descending order.
    """

    assert len(y_score) == len(y_true), 'Lengths of ground-truth and prediction arrays do not match.'
    sorted_y_score, sorted_y_true = synchronize_sort(y_score, y_true)

    if num_bins:
        binned_array = np.array_split(sorted_y_score, num_bins)
        sorted_y_score = np.array([i[0] for i in binned_array])
    else:
        binned_array = None

    results, counts = _generate_performance_values(y_true, sorted_y_true, metric, num_bins, binned_array)

    return PerformanceCurve(results, counts, sorted_y_score, metric)


def random_performance_curve(
        y_true: np.ndarray,
        metric: Metric,
        num_bins: Optional[int] = None
) -> PerformanceCurveLike:

    idx = list(range(len(y_true)))
    random.shuffle(idx)
    random_y_true = np.array([y_true[i] for i in idx])
    binned_array = np.array_split(random_y_true, num_bins) if num_bins else None

    results, counts = _generate_performance_values(y_true, random_y_true, metric, num_bins, binned_array)

    return PerformanceCurveLike(results, counts, metric)


def perfect_performance_curve(
        y_true: np.ndarray,
        metric: Metric,
        num_bins: Optional[int] = None
) -> PerformanceCurveLike:

    perfect_y_true = np.sort(y_true)[::-1]
    binned_array = np.array_split(perfect_y_true, num_bins) if num_bins else None

    results, counts = _generate_performance_values(y_true, perfect_y_true, metric, num_bins, binned_array)

    return PerformanceCurveLike(results, counts, metric)
