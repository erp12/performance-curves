# -*- coding: utf-8 -*-
"""Utility for working with a collection of performance curves."""
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from performance_curves.metric import MetricInfo
from performance_curves.performance_curve import (
    PerformanceCurve,
    RandomPerformanceCurve,
    PerfectPerformanceCurve,
)


class PerformanceCurveBundle:
    """A bundle of multiple performance curves.

    Each curve represents one of a variety of methods (e.g. ranking systems, or ML models) applied to the same test
    data and evaluated under the same metric.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        method_scores: Mapping[str, np.ndarray],
        metric: MetricInfo,
        num_bins: Optional[int] = None,
        order_descending: bool = True,
        make_perfect: bool = True,
        make_random: bool = True,
        num_trials: int = 1,
    ):
        """Instantiate a `PerformanceCurveBundle`.

        Args:
            y_true: Ground truth labels.
            method_scores: Scores from collection of methods being compared.
            metric: Metric to compare with.
            num_bins: Number of bins to partition the scores into.
            order_descending: If True, assume x-axis is descending.
            make_perfect: If True, include the hypothetical perfect ranking in the comparison.
            make_random: If True, include a random ranking in the comparison.
            num_trials: If including the random ranking, the random rankings that will be averaged.
        """
        self.y_true = y_true
        self.method_scores = method_scores
        self.metric = metric
        self.num_bins = num_bins
        self.order_descending = order_descending

        self.curves = dict()
        self.random_curve = None
        self.perfect_curve = None

        for method_name, y_score in self.method_scores.items():
            self.curves[method_name] = PerformanceCurve(
                self.y_true, y_score, self.metric, self.num_bins, self.order_descending
            )

        if make_perfect:
            self.perfect_curve = PerfectPerformanceCurve(self.y_true, self.metric, self.num_bins)
        if make_random:
            self.random_curve = RandomPerformanceCurve(self.y_true, self.metric, num_trials, self.num_bins)

    def plot(self, percentage_x=False):
        """Plot the bundle of curves.

        Args:
            percentage_x:  If True, put the percentage of data cases on the X axis, otherwise use absolute counts.

        """

        def create_x_axis_values(raw_counts: np.ndarray):
            x_values = raw_counts / raw_counts[-1] if percentage_x else raw_counts
            return x_values

        fig, ax = plt.subplots()
        for curve_name, curve in self.curves.items():
            ax.plot(
                create_x_axis_values(curve.case_counts),
                curve.performance_values,
                label=curve_name,
            )
        if self.random_curve:
            ax.plot(
                create_x_axis_values(self.random_curve.case_counts),
                self.random_curve.performance_values,
                label="random",
                linestyle="dashed",
                color="r",
            )
        if self.perfect_curve:
            ax.plot(
                create_x_axis_values(self.perfect_curve.case_counts),
                self.perfect_curve.performance_values,
                label="perfect",
                linestyle="dashdot",
                color="k",
            )
        ax.set_xlabel("Percentage of Total Cases" if percentage_x else "Number of Cases")
        ax.set_ylabel(self.metric.name)
        plt.legend()
        plt.show()
