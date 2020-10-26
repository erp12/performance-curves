import operator
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional

from performance_curves.utils import synchronize_sort


class PerformanceCurveLike:
    """Stores values of the x- and y-axis of the performance curve and allows plotting"""
    def __init__(self, metric_value, example_counts):
        self.metric_value = metric_value
        self.example_counts = example_counts

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.example_counts, self.metric_value)
        ax.set_xlabel('Number of Evaluated Cases Ranked in Descending Order')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Curve')
        plt.show()


class PerformanceCurve(PerformanceCurveLike):
    def __init__(self, metric_value, example_counts, score_cutoff):
        super().__init__(metric_value, example_counts)
        self.score_cutoff = score_cutoff

    def plot_with_cutoff(self):
        fig, ax = plt.subplots()
        ax.plot(self.score_cutoff, self.metric_value)
        ax.set_xlabel('Probability Estimates in Descending Order')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Curve')
        plt.show()

    def cutoff_at(
            self,
            performance_point: float,
            comparison_operator: str,
            count_upper_bound: Optional[int] = None,
            count_lower_bound: Optional[int] = None,
            cutoff_upper_bound: Optional[float] = None,
            cutoff_lower_bound: Optional[float] = None,
            highest: bool = False
    ) -> float:
        """
        Returns score cutoff which meets a certain performance criteria given other optional counts and scores
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
            cutoff_upper_bound: float, default = None
                Upper bound of acceptable score cutoffs
            cutoff_lower_bound: float, default = None
                Lower bound of acceptable score cutoffs
            highest: bool, default = False
                If True, return the highest score cutoff in case there are multiple values that satisfy all the
                constraints. Otherwise, return the lowest score cutoff.

        Return:
            The single highest/lowest score cutoff that satisfies all constraints if exists. Otherwise, return None.

        Examples:
            `cutoff_at(0.80, '>=', count_lower_bound=100)` returns the lowest score cutoff that produces a performance
            of at least 0.80 and uses at least 100 observations.
            `cutoff_at(0.75, '<', cutoff_upper_bound=0.25, highest=True)` returns the highest score cutoff less than
            0.25 that produces a performance less than 0.75.
        """

        assert len(self.metric_value) == len(self.example_counts), \
            'Lengths of performance values array and counts array do not match.'
        assert len(self.metric_value) == len(self.score_cutoff), \
            'Lengths of performance values array and probability estimates array do not match.'

        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '==': operator.eq}
        if comparison_operator in ops:
            metric_ind = np.where(ops[comparison_operator](self.metric_value, performance_point))[0]
        else:
            raise ValueError("comparison_operator argument could only take one of the following strings: "
                             "'>', '<', '>=', '<=', '=='.")

        bound_ind = np.array([i for i in range(len(self.metric_value))])
        if count_upper_bound is not None:
            count_upper_bound_ind = np.where(self.example_counts <= count_upper_bound)[0]
            bound_ind = np.intersect1d(bound_ind, count_upper_bound_ind)
        if count_lower_bound is not None:
            count_lower_bound_ind = np.where(self.example_counts >= count_lower_bound)[0]
            bound_ind = np.intersect1d(bound_ind, count_lower_bound_ind)
        if cutoff_upper_bound is not None:
            cutoff_upper_bound_ind = np.where(self.score_cutoff <= cutoff_upper_bound)[0]
            bound_ind = np.intersect1d(bound_ind, cutoff_upper_bound_ind)
        if cutoff_lower_bound is not None:
            cutoff_lower_bound_ind = np.where(self.score_cutoff >= cutoff_lower_bound)[0]
            bound_ind = np.intersect1d(bound_ind, cutoff_lower_bound_ind)

        result_ind = np.intersect1d(metric_ind, bound_ind)
        if len(result_ind) > 0:
            if highest:
                return self.score_cutoff[result_ind[0]]
            else:
                return self.score_cutoff[result_ind[-1]]

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
        results: array of float, shape = [n_samples] or [n_bins]
            Performance values of metrics of interest when setting the score cutoff at the corresponding y_score
        counts: array of int, shape = [n_samples] or [n_bins]
            Number of cases corresponding with the same position's y_score or metric_value
        sorted_y_score: array of float, shape = [n_samples] or [n_bins]
            In case num_bins is `None`, return probability estimates of all data points sorted in descending order.
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
