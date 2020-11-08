from typing import Callable
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Metric:
    def __init__(self,
                 name: str,
                 func: Callable[[np.ndarray, np.ndarray], float],
                 should_minimize: bool = False):
        self.name = name
        self.func = func
        self.should_minimize = should_minimize


class PrecisionMetric(Metric):
    def __init__(self, name='Precision', func=precision_score, should_minimize=False):
        super().__init__(name, func, should_minimize)


class RecallMetric(Metric):
    def __init__(self, name='Recall', func=recall_score, should_minimize=False):
        super().__init__(name, func, should_minimize)


class F1Metric(Metric):
    def __init__(self, name='F1 Score', func=f1_score, should_minimize=False):
        super().__init__(name, func, should_minimize)


class AccuracyMetric(Metric):
    def __init__(self, name='Accuracy', func=accuracy_score, should_minimize=False):
        super().__init__(name, func, should_minimize)
