from typing import Callable
import numpy as np


# TODO: write custom metrics for precision, recall, F1 and accuracy
class Metric:
    def __init__(self,
                 name: str,
                 func: Callable[[np.ndarray, np.ndarray], float],
                 should_minimize: bool = False):
        self.name = name
        self.func = func
        self.should_minimize = should_minimize
