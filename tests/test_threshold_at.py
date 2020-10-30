import numpy as np
import pytest

from performance_curves.performance_curve import performance_curve
from sklearn.metrics import recall_score


def test_threshold_at():
    #
    # Givens
    #

    # two arrays of ground truth and predicted scores, correspondingly
    y_true = np.array([0, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.91, 0.8, 0.05])

    #
    # Whens
    #

    x = performance_curve(y_true, y_score, recall_score)

    # smallest threshold to achieve at least a 0.80 recall score
    desired_threshold1 = x.threshold_at(0.80, '>=')
    # highest threshold to achieve at most a 0.50 recall score with a minimum of 3 cases
    desired_threshold2 = x.threshold_at(0.50, '<=', count_lower_bound=3, highest=True)


    #
    # Thens
    #
    assert desired_threshold1 == 0.05
    assert desired_threshold2 is None
    with pytest.raises(ValueError):
        x.threshold_at(0.80, '!=')
