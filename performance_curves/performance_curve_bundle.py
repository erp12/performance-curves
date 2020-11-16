from performance_curves.performance_curve import *


class PerformanceCurveBundle:
    def __init__(self,
                 y_true: np.ndarray,
                 y_scores: List[np.ndarray],
                 metric: Metric,
                 model_names: List[str],
                 num_bins: Optional[int] = None,
                 order_descending: bool = True,
                 random_model: bool = True,
                 random_seed: Optional[int] = None,
                 num_trials: Optional[int] = 1,
                 perfect_model: bool = True):
        self.y_true = y_true
        self.y_scores = y_scores
        self.metric = metric
        self.model_names = model_names
        self.num_bins = num_bins

        score_sizes = [len(y_score) for y_score in self.y_scores]
        assert all(score_size == score_sizes[0] for score_size in score_sizes), \
            "Number of predictions outputted by each model must be the same."

        all_performance_values = []
        all_case_counts = []
        for i in range(len(self.y_scores)):
            individual_curve = PerformanceCurve(self.y_true, y_scores[i], self.metric, self.num_bins,
                                                order_descending)
            all_performance_values.append(individual_curve.performance_values)
            all_case_counts.append(individual_curve.case_counts)
        self.bundle = dict(zip(self.model_names, zip(all_performance_values, all_case_counts)))

        if random_model:
            random_curve = RandomPerformanceCurve(self.y_true, self.metric, num_trials, self.num_bins, random_seed)
            self.bundle["random_model"] = tuple(zip(random_curve.performance_values, random_curve.case_counts))
        if perfect_model:
            perfect_curve = PerfectPerformanceCurve(self.y_true, self.metric, self.num_bins)
            self.bundle["perfect_model"] = tuple(zip(perfect_curve.performance_values, perfect_curve.case_counts))

    def plot(self):
        fig, ax = plt.subplots()
        for key, value in self.bundle.items():
            if key == "random_model":
                ax.plot(value[1], value[0], label=key, linestyle='dashed', color='r')
            elif key == "perfect_model":
                ax.plot(value[1], value[0], label=key, linestyle='dashdot', color='k')
            else:
                ax.plot(value[1], value[0], label=key)
        ax.set_xlabel('Number of Cases')
        ax.set_ylabel(self.metric.name)
        plt.legend()
        plt.show()
