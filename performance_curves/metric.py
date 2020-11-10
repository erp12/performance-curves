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


PRECISION = Metric("Precision", precision_score)
RECALL = Metric("Recall", recall_score)
F1 = Metric("F1 Score", f1_score)
ACCURACY = Metric("Accuracy", accuracy_score)
