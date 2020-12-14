# Performance Curves

## Rationale 

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
- [ ] Method to select best model for a given objective

### Questions (need to be revisited)
- Do we want to allow multiple metrics in the same `performance_curve` function?
    - Pro: save computation time, since given the same model, no matter what the metric is, we still use the same `y_pred` and `y_score`
- Do we want to allow multiple models in the same `performance_curve` function?
    - Con: doesn't save any computation time (cause there's nothing to be reused across models)
    
