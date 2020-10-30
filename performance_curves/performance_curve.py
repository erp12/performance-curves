import operator
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional

from performance_curves.utils import synchronize_sort


class PerformanceCurveLike:
    """Stores values of the x- and y-axis of the performance curve and allows plotting"""
    def __init__(self, performance_values, case_counts):
        self.performance_values = performance_values
        self.case_counts = case_counts

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.case_counts, self.performance_values)
        ax.set_xlabel('Number of Cases Ranked in Descending Order')
        ax.set_ylabel('Performance Value')
        ax.set_title('Performance Curve')
        plt.show()


class PerformanceCurve(PerformanceCurveLike):
    def __init__(self, performance_values, case_counts, score_thresholds):
        super().__init__(performance_values, case_counts)
        self.score_thresholds = score_thresholds

    def plot_with_cutoff(self):
        fig, ax = plt.subplots()
        ax.plot(self.score_thresholds, self.performance_values)
        ax.set_xlabel('Predicted Score (Probability Estimate) in Descending Order')
        ax.set_ylabel('Performance Value')
        ax.set_title('Performance Curve')
        plt.show()

    def threshold_at(
            self,
            performance_point: float,
            comparison_operator: str,
            count_upper_bound: Optional[int] = None,
            count_lower_bound: Optional[int] = None,
            threshold_upper_bound: Optional[float] = None,
            threshold_lower_bound: Optional[float] = None,
            highest: bool = False
    ) -> float:
        """
        Returns score threshold which meets a certain performance criteria given other optional counts and scores
        constraints

        Arguments:
            performance_point: float
                Baseline performance value to compare against
            comparison_operator: str, valid options include '>', '<', '>=', '<=', and '=='
                Operator type used to compare the actual performance values and user's desired performance baseline
            count_upper_bound: int, default = None
                Maximum number of observations required to achieve the performance
            count_lower_bound: int, default = None
                Minimum number of observations required to achieve the performance
            threshold_upper_bound: float, default = None
                Upper bound of acceptable score thresholds
            threshold_lower_bound: float, default = None
                Lower bound of acceptable score thresholds
            highest: bool, default = False
                If True, return the highest score threshold in case there are multiple values that satisfy all the
                constraints. Otherwise, return the lowest score threshold.

        Return:
            The single highest/lowest score threshold that satisfies all constraints if exists. Otherwise, return None.

        Examples:
            `threshold_at(0.80, '>=', count_lower_bound=100)` returns the lowest score threshold that produces a
            performance of at least 0.80 and uses at least 100 observations.
            `threshold_at(0.75, '<', threshold_upper_bound=0.25, highest=True)` returns the highest score threshold less
            than 0.25 that produces a performance value less than 0.75.
        """

        assert len(self.performance_values) == len(self.case_counts), \
            'Lengths of performance values array and counts array do not match.'
        assert len(self.performance_values) == len(self.score_thresholds), \
            'Lengths of performance values array and thresholds array do not match.'

        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '==': operator.eq}
        if comparison_operator in ops:
            metric_ind = np.where(ops[comparison_operator](self.performance_values, performance_point))[0]
        else:
            raise ValueError("comparison_operator argument could only take one of the following strings: "
                             "'>', '<', '>=', '<=', '=='.")

        bound_ind = np.array([i for i in range(len(self.performance_values))])
        if count_upper_bound is not None:
            count_upper_bound_ind = np.where(self.case_counts <= count_upper_bound)[0]
            bound_ind = np.intersect1d(bound_ind, count_upper_bound_ind)
        if count_lower_bound is not None:
            count_lower_bound_ind = np.where(self.case_counts >= count_lower_bound)[0]
            bound_ind = np.intersect1d(bound_ind, count_lower_bound_ind)
        if threshold_upper_bound is not None:
            threshold_upper_bound_ind = np.where(self.score_thresholds <= threshold_upper_bound)[0]
            bound_ind = np.intersect1d(bound_ind, threshold_upper_bound_ind)
        if threshold_lower_bound is not None:
            threshold_lower_bound_ind = np.where(self.score_thresholds >= threshold_lower_bound)[0]
            bound_ind = np.intersect1d(bound_ind, threshold_lower_bound_ind)

        result_ind = np.intersect1d(metric_ind, bound_ind)
        if len(result_ind) > 0:
            if highest:
                return self.score_thresholds[result_ind[0]]
            else:
                return self.score_thresholds[result_ind[-1]]

        return None


class RandomPerformanceCurve(PerformanceCurveLike):
    pass


class PerfectPerformanceCurve(PerformanceCurveLike):
    pass


def performance_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
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
        metric: callable method(s)
            Function(s) that assess a classifier's prediction error for specific purposes given ground truth and
            prediction (e.g.: sklearn.metrics.recall_score, sklearn.metrics.precision_score, etc).
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

    return PerformanceCurve(np.array(results), np.array(counts), sorted_y_score)
