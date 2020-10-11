import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional

from performance_curves.utils import synchronize_sort


class PerformanceCurve:
    """Stores values of the x- and y-axis of the performance curve and allows plotting"""
    def __init__(self, cutoff, metric_value, example_counts):
        self.cutoff = cutoff
        self.metric_value = metric_value
        self.example_counts = example_counts

    def plot_with_thresholds(self):
        fig, ax = plt.subplots()
        ax.plot(self.cutoff, self.metric_value)
        ax.set_xlabel('Probability Estimates in Descending Order')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Curve')
        plt.show()

    def plot_with_counts(self):
        fig, ax = plt.subplots()
        ax.plot(self.example_counts, self.metric_value)
        ax.set_xlabel('Number of Evaluated Cases Ranked in Descending Order')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Curve')
        plt.show()


def performance_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    num_bins: Optional[int] = None
) -> PerformanceCurve:
    """
    Computes performance value of a metric of interest at each probability estimate (or each upper bound of a range of
    probability estimates in case of binning) given that they are sorted in descending order

    Arguments:
        y_true: 1d-array like
            Ground truth (correct) labels.
        y_score: 1d array-like
            Probability estimates as returned by a classifier.
        metric: callable method(s)
            Function(s) that assess a classifier's prediction error for specific purposes given ground truth and
            prediction (e.g.: sklearn.metrics.recall_score, sklearn.metrics.precision_score, etc).
        num_bins: int, optional (default=None)
            Number of bins (or bucket) to divide the range of values into.

    Return:
        sorted_y_score: array of float, shape = [n_samples] or [n_bins]
            In case num_bins is `None`, return probability estimates of all data points sorted in descending order.
            In case num_bins is not `None`, return upper bounds of the bins sorted in descending order.
        results: array of float, shape = [n_samples] or [n_bins]
            Performance values of metrics of interest when setting the cutoff at the corresponding y_score
        counts: array of int, shape = [n_samples] or [n_bins]
            Number of cases corresponding with the same position's y_score or metric_value
    """

    assert len(y_score) == len(y_true), 'Lengths of ground-truth and prediction arrays do not match.'

    size = len(y_true)
    result_size = num_bins if num_bins else size
    y_pred = np.zeros(size)
    results = list()
    counts = list()
    current_count = 0

    sorted_y_score, sorted_y_true = synchronize_sort(y_score, y_true)

    if num_bins:
        binned_array = np.array_split(sorted_y_score, num_bins)
        sorted_y_score = np.array([i[0] for i in binned_array])

    for i in range(result_size):
        if num_bins:
            num_elements = len(binned_array[i])
            first_zero = (y_pred == 0).argmax()
            y_pred[first_zero:(first_zero + num_elements)] = 1
            current_count += num_elements
        else:
            y_pred[i] = 1
            current_count += 1
        results.append(metric(sorted_y_true, y_pred))
        counts.append(current_count)

    return PerformanceCurve(sorted_y_score, np.array(results), np.array(counts))
