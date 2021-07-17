# -*- coding: utf-8 -*-
"""Collection of utilities designed to model metadata about a metric used to evaluate models.

For the purposes of the `performance_curves` package, a metric is any function that takes two arguments:

1. A 1-dimensional numpy array of model scores
2. A 1-dimensional numpy array of ground-truth labels

and returns a single numeric value that correlates (positively or negatively) with certain aspect of a
model's performance.

There are a variety of common metrics that are used generically across the ML community. `performance_curve` provides
`MetricInfo` objects for some of these metrics.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


@dataclass
class MetricInfo:
    # noinspection PyUnresolvedReferences
    """Metadata associated with a metric function.

    Attributes:
        name: The name of the metric.
        metric: The metric function.
        should_minimize: Indicates if the metric should be minimized (`True`) or maximized (`False`).
    """
    name: str
    metric: Callable[[np.ndarray, np.ndarray], float]
    should_minimize: bool = False


PRECISION = MetricInfo(name="c", metric=precision_score)
"""The proportion of positive predictions that correspond to true positive data cases or `tp / (tp + fp)`.
See the [scikit-learn documentation.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

Note: This is a mutable global variable. Beware that in-place operations will be shared by all consumers.
"""

RECALL = MetricInfo(name="Recall", metric=recall_score)
"""The proportion of true positive data cases given a positive predicted class or `tp / (tp + fn)`.
See the [scikit-learn documentation.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

Note: This is a mutable global variable. Beware that in-place operations will be shared by all consumers.
"""

F1 = MetricInfo(name="F1 Score", metric=f1_score)
"""The the harmonic mean of the precision an recall, also known as balanced F-score or F-measure.
See the [scikit-learn documentation.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

Note: This is a mutable global variable. Beware that in-place operations will be shared by all consumers.
"""

ACCURACY = MetricInfo(name="Accuracy", metric=accuracy_score)
"""The proportion of correct classifications.
See the [scikit-learn documentation.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

Note: This is a mutable global variable. Beware that in-place operations will be shared by all consumers.
"""
