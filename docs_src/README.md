# Performance Curves

Performance curve provides visual representations for a multitude of evaluation metrics for binary classification models over a range of possible operating conditions. 

What performance curve requires:
- Every data case has a binary ground truth label
- Every data case can be ordered relative to each other
  - Probability, statistical model score, rank, points assigned by rule system, etc!
- An evaluation/metric function

What performance curve offers:
- Overview of a method's performance across a range of decision boundaries
- An intuitive visualization to address the early retrieval problem, where the correctness of only a small number of highest-ranked predictions is of interest
- No required knowledge of the relative misclassification costs
- Robustness against imbalanced data problem  
- Means to compare models to each other, as well as with random and perfect classifiers over a range of operating conditions on a single dataset


## Rationale

### Problem 1: 

There are many popular metrics, such as accuracy, precision, recall and F-measure, that are sensitive to a good choice of threshold that separates between positive and negative classes. While any such threshold can be reasonable in a particular application, it is not obvious how the right threshold value should be chosen or if a threshold should be chosen at all! For example, the tradeoff between sensitivity and specificity is closely related to the costs of Type I and Type II errors. Often times, based on business requirements, we have a vague sense of which type of error is more costly, thus knowing which metrics between sensitivity and specificity the model should optimize over. Still, these costs are rarely quantifiable. By choosing one single threshold for the predictive evaluation metric to rely on, we incorrectly convey that the relative costs of mis-classification errors can and have been quantified. 

*Solution*: 

Use threshold-free measures! Performance curve provides a visual representation of the range of performance with no knowledge of misclassification costs or threshold value, and with that a higher level of utility. 

### Problem 2: 

Most of the widely used ranking metrics (i.e. metrics based on how well a method ranks data cases), such as area under receiver operating characteristic (AUROC) curve, or precision-recall (PR) curve, do not explicitly translate to business requirements. For example, given limited human capacity in reviewing positive predictions of a statistical method, neither AUROC nor PR curves indicate the method's performance when only a small number of positively predicted cases get investigated. Similarly, given a fixed precision or recall, both metrics do not describe the exact number of data cases that must be reviewed to achieve the desired performance. Additionally, both ROC and PR curves omit the specific threshold information from the graphic, therefore being irrelevant for choosing a threshold value in case threshold is indeed necessary.

*Solution*: 

### Problem 3: 

AUROC or AUPRC or any scalar performance measures gives an apparently definitive answer to which method is better. However, it often happens that one method is superior to another in some circumstances and inferior in others, and these widely used scalar performance measures give no assistance in identifying the circumstances in which a particular method is superior.

*Solution*:
@TODO: create an example of 2 models on a dataset where model B's AUROC > model A's AUROC but we want to use model A's instead since it addresses the early retrieval problem better.

### Problem 4: 
@TODO: example of user-defined metric over a range of operating conditions

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
- [ ] Method to select best model for a given objective
- [ ] Linter in unit tests
- [ ] Account for the variability of the model with respect to the metrics

### Questions
- Do we want to allow multiple metrics in the same `performance_curve` function?
    - Pro: save computation time, since given the same model, no matter what the metric is, we still use the same `y_pred` and `y_score`
- Do we want to allow multiple models in the same `performance_curve` function?
    - Con: doesn't save any computation time (cause there's nothing to be reused across models)
  
