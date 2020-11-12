import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

from performance_curves.utils import synchronize_sort, get_bin_sizes
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

    def plot(self, target_performance_values: Optional[List[float]] = None):
        if target_performance_values is not None and not isinstance(self, PerformanceCurve):
            raise ValueError("Cannot annotate target performance values of " + type(self).__name__)
        fig, ax = plt.subplots()
        ax.plot(self.case_counts, self.performance_values)
        if not target_performance_values:
            # @todo Put annotations at target_performance_values using self.threshold_at
            pass
        ax.set_xlabel("Number of Cases")
        ax.set_ylabel(self.metric.name)
        plt.show()


class NonRandomPerformanceCurve(PerformanceCurveLike):
    def __init__(self,
                 rearranged_y_true: np.ndarray,
                 metric: Metric,
                 num_bins: Optional[int] = None):
        self.rearranged_y_true = rearranged_y_true
        self.metric = metric
        self.num_bins = num_bins
        bin_sizes = get_bin_sizes(rearranged_y_true, num_bins) if num_bins else None
        performance_values, case_counts = _generate_performance_values(rearranged_y_true, metric, bin_sizes)
        super().__init__(performance_values, case_counts, metric)


class PerformanceCurve(NonRandomPerformanceCurve):
    def __init__(self,
                 y_true: np.ndarray,
                 y_score: np.ndarray,
                 metric: Metric,
                 num_bins: Optional[int] = None,
                 order_descending: bool = True):
        self.y_true = y_true
        self.y_score = y_score
        assert len(self.y_score) == len(self.y_true), "Lengths of ground-truth and prediction arrays do not match."
        sorted_y_score, sorted_y_true = synchronize_sort(self.y_score, self.y_true, descending=order_descending)
        super().__init__(sorted_y_true, metric, num_bins)

    def threshold_at(self, performance_point: float) -> List[Tuple[float, int]]:
        pass


class PerfectPerformanceCurve(NonRandomPerformanceCurve):
    def __init__(self,
                 y_true: np.ndarray,
                 metric: Metric,
                 num_bins: Optional[int] = None):
        self.y_true = y_true
        perfect_y_true = np.sort(y_true)[::-1]
        super().__init__(perfect_y_true, metric, num_bins)


class RandomPerformanceCurve(PerformanceCurveLike):
    def __init__(self,
                 y_true: np.ndarray,
                 metric: Metric,
                 num_trials: int = 1,
                 num_bins: Optional[int] = None,
                 random_seed: Optional[int] = None):
        self.y_true = y_true
        self.num_trials = num_trials
        self.num_bins = num_bins
        idxs = np.arange(len(self.y_true))
        bin_sizes = get_bin_sizes(self.y_true, self.num_bins) if self.num_bins else None

        trials = []
        case_counts = None
        if random_seed:
            np.random.seed(random_seed)
        for _ in range(self.num_trials):
            np.random.shuffle(idxs)
            random_y_true = np.array([self.y_true[i] for i in idxs])
            performance_values, case_counts = _generate_performance_values(random_y_true, metric, bin_sizes)
            trials.append(performance_values)

        performance_values = np.average(trials, axis=0)
        super().__init__(performance_values, case_counts, metric)


def _generate_performance_values(
        rearranged_y_true: np.ndarray,
        metric: Metric,
        bin_sizes: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
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
            y_pred[first_zero:(first_zero + num_elements)] = 1
            current_count += num_elements
        else:
            y_pred[i] = 1
            current_count += 1
        performance_values.append(metric.func(rearranged_y_true, y_pred))
        case_counts.append(current_count)
    return np.array(performance_values), np.array(case_counts)
