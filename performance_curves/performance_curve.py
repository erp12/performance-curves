"""The primary module for constructing and transforming performance curves."""
import warnings
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from performance_curves.metric import MetricInfo
from performance_curves._utils import synchronize_sort, get_bin_sizes


class PerformanceCurveLike:
    """Stores values of the x- and y-axis of the performance curve and allows plotting."""

    def __init__(
        self,
        performance_values: np.ndarray,
        case_counts: np.ndarray,
        metric: MetricInfo,
    ):
        """Instantiate a `PerformanceCurveLike`."""
        self.performance_values = performance_values
        self.case_counts = case_counts
        self.metric = metric

    def plot(
        self,
        percentage_x=False,
        target_performance_values: Optional[List[float]] = None,
    ):
        """Plot the performance curve."""
        if target_performance_values is not None and not isinstance(self, PerformanceCurve):
            raise ValueError("Cannot annotate target performance values of " + type(self).__name__)
        fig, ax = plt.subplots()
        ax.plot(
            self.case_counts / self.case_counts[-1] if percentage_x else self.case_counts,
            self.performance_values,
        )
        ax.set_xlabel("Percentage of Total Cases" if percentage_x else "Number of Cases")
        ax.set_ylabel(self.metric.name)
        if not target_performance_values:
            # @todo Put annotations at target_performance_values using self.threshold_at
            pass
        plt.show()


class NonRandomPerformanceCurve(PerformanceCurveLike):
    """Base class for non-random performance curves."""

    def __init__(
        self,
        rearranged_y_true: np.ndarray,
        metric: MetricInfo,
        num_bins: Optional[int] = None,
    ):
        """Instantiate a `NonRandomPerformanceCurve`."""
        self.rearranged_y_true = rearranged_y_true
        self.metric = metric
        self.num_bins = num_bins
        bin_sizes = get_bin_sizes(rearranged_y_true, num_bins) if num_bins else None
        performance_values, case_counts = _generate_performance_values(rearranged_y_true, metric, bin_sizes)
        super().__init__(performance_values, case_counts, metric)


class PerformanceCurve(NonRandomPerformanceCurve):
    """Basic performance curve type corresponding to a ranking method/model."""

    def __init__(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        metric: MetricInfo,
        num_bins: Optional[int] = None,
        order_descending: bool = True,
    ):
        """Instantiate a `PerformanceCurve`."""
        self.y_true = y_true
        self.y_score = y_score
        self.order_descending = order_descending
        assert len(self.y_score) == len(self.y_true), "Lengths of ground-truth and prediction arrays do not match."
        sorted_y_score, sorted_y_true = synchronize_sort(self.y_score, self.y_true, descending=self.order_descending)
        super().__init__(sorted_y_true, metric, num_bins)


class PerfectPerformanceCurve(NonRandomPerformanceCurve):
    """A performance curve corresponding to a hypothetical perfect ranking method."""

    def __init__(self, y_true: np.ndarray, metric: MetricInfo, num_bins: Optional[int] = None):
        """Instantiate a `PerfectPerformanceCurve`."""
        self.y_true = y_true
        perfect_y_true = np.sort(y_true)[::-1]
        super().__init__(perfect_y_true, metric, num_bins)


class RandomPerformanceCurve(PerformanceCurveLike):
    """A performance curve corresponding to the average performance of a random ranking method across many trials."""

    def __init__(
        self,
        y_true: np.ndarray,
        metric: MetricInfo,
        num_trials: int = 1,
        num_bins: Optional[int] = None,
    ):
        """Instantiate a `RandomPerformanceCurve`."""
        self.y_true = y_true
        self.num_trials = num_trials
        self.num_bins = num_bins

        idxs = np.arange(len(self.y_true))
        bin_sizes = get_bin_sizes(self.y_true, self.num_bins) if self.num_bins else None

        trials = []
        case_counts = None
        for _ in range(self.num_trials):
            np.random.shuffle(idxs)
            random_y_true = np.array([self.y_true[i] for i in idxs])
            performance_values, case_counts = _generate_performance_values(random_y_true, metric, bin_sizes)
            trials.append(performance_values)

        performance_values = np.average(trials, axis=0)
        super().__init__(performance_values, case_counts, metric)


def _generate_performance_values(
    rearranged_y_true: np.ndarray,
    metric: MetricInfo,
    bin_sizes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if bin_sizes is None and len(rearranged_y_true) >= 10000:
        warnings.warn(
            "Given the number of cases you have, consider turning `num_bins` on to reduce runtime.",
            RuntimeWarning,
        )

    size = len(rearranged_y_true)
    result_size = len(bin_sizes) if bin_sizes else size
    y_pred = np.zeros(size)
    current_count = 0
    performance_values = list()
    case_counts = list()

    for i in range(result_size):
        if bin_sizes:
            num_elements = bin_sizes[i]
            first_zero = (y_pred == 0).argmax()
            y_pred[first_zero : (first_zero + num_elements)] = 1
            current_count += num_elements
        else:
            y_pred[i] = 1
            current_count += 1
        performance_values.append(metric.metric(rearranged_y_true, y_pred))
        case_counts.append(current_count)
    return np.array(performance_values), np.array(case_counts)
