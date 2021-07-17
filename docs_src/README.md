# Performance Curves

Performance curve provides visual representations for a multitude of evaluation metrics for binary classification models over a range of possible operating conditions. 

Key properties that a performance curve could satisfy: 
- Give an overview of a range of performances a model could achieve with varying thresholds or varying percentages of predictions taken into account. The model performance could be measured with traditional threshold metrics (e.g. precision, recall, F-1 score, etc), with ranking metrics (e.g. ROC, precision-recall curve), or with any user-custom metrics that depend on the ordering of instances and not the actual predicted values, thus requiring no threshold cutoff. 
- Is a diagnostic tool that allows users to compare the model’s performance on high-ranked instances with low-ranked instances, thereby addressing the early retrieval problem, where the correctness of only a small number of highest-ranked predictions is of interest
- Provide means to compare models to each other, as well as with random and perfect classifiers over a range of operating conditions on a single dataset

## Rationale

There are two most common families of evaluation metrics used in the context of classification: threshold metrics (e.g. accuracy, precision, recall, F-measure) and ranking metrics (e.g. receiver operating characteristic (ROC) curve, precision-recall (PR) curve). Although threshold metrics are intuitive and easy to calculate, they assume full knowledge of the class distributions present in the dataset. Specifically, they require a threshold cutoff to separate a dataset into positively and negatively predicted classes. Although this can be reasonable in a particular application, it is not obvious how the right threshold value should be chosen. On the other hand, although popular existing ranking metrics such as ROC and PR curves don’t make any assumptions about class distributions and evaluate classifiers over variable thresholds, they are less intuitive to non-technical audiences and do not explicitly translate to actionable business requirements. 
This is where a performance curve comes into play, attempting to combine the best of both worlds to provide threshold-free visual evaluations for binary classifiers based on any traditional threshold metrics or any user-defined rank-based metrics. 

## Getting Started

## Development

This project uses `tox` to run tests and perform basic code management. Using a python 3 environment, install `tox`
using the following command.

```
pip install tox
```

### Running Tests

To run the unit test suite for the package, invoke `tox` using the following command. You can change the minor version
of Python using the one of the variants.

```
tox -e test-py36  # or test-py37 or test-py38
```

### TODO:
- [ ] Rewrite developer guide
- [ ] Contributing guide
- [ ] Release notes
- [ ] Examples
- [ ] Create performance curve bundle (1 test set / 1 metric / multiple models)
- [ ] Method to select best model for a given objective
- [ ] Linter in unit tests
- [ ] Account for the variability of the model with respect to the metrics

### Questions
- Do we want to allow multiple metrics in the same `performance_curve` function?
    - Pro: save computation time, since given the same model, no matter what the metric is, we still use the same `y_pred` and `y_score`
- Do we want to allow multiple models in the same `performance_curve` function?
    - Con: doesn't save any computation time (cause there's nothing to be reused across models)
  
