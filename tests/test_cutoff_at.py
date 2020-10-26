import numpy as np

from performance_curves.performance_curve import performance_curve
from sklearn.metrics import recall_score


def test_cutoff_at():
    #
    # Givens
    #

    # two arrays of ground truth and probability estimates, correspondingly
    y_true = np.array([0, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.91, 0.8, 0.05])

    #
    # Whens
    #

    x = performance_curve(y_true, y_score, recall_score)

    # smallest cutoff to achieve at least a 0.80 recall score
    desired_cutoff1 = x.cutoff_at(0.80, '>=')
    # highest cutoff to achieve at most a 0.50 recall score with a minimum of 3 observations
    desired_cutoff2 = x.cutoff_at(0.50, '<=', count_lower_bound=3, highest=True)


    #
    # Thens
    #
    assert desired_cutoff1 == 0.05
    assert desired_cutoff2 is None
    with pytest.raises(ValueError):
        x.cutoff_at(0.80, '!=')
