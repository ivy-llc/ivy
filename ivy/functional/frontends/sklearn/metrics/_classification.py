import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from sklearn.utils.multiclass import type_of_target


@to_ivy_arrays_and_back
def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    # TODO: implement sample_weight
    y_type = type_of_target(y_true)
    if y_type.startswith("multilabel"):
        diff_labels = ivy.count_nonzero(y_true - y_pred, axis=1)
        ret = ivy.equal(diff_labels, 0).astype("int64")
    else:
        ret = ivy.equal(y_true, y_pred).astype("int64")
    ret = ret.sum().astype("int64")
    if normalize:
        ret = ret / y_true.shape[0]
        ret = ret.astype("float64")
    return ret


@to_ivy_arrays_and_back
def precision_score(y_true, y_pred, *, average="binary", sample_weight=None):
    # TODO: implement sample_weight
    y_type = type_of_target(y_true)
    if y_type.startswith("multilabel"):
        true_positives = ivy.count_nonzero(
            ivy.equal(y_true, y_pred).astype("int64"), axis=0
        )
        all_positives = ivy.count_nonzero(y_pred, axis=0)
    else:
        true_positives = ivy.count_nonzero(
            ivy.equal(y_true, y_pred).astype("int64"), axis=1
        )
        all_positives = ivy.count_nonzero(y_pred)
    if average == "binary":
        precision = true_positives / all_positives
    elif average == "micro":
        precision = ivy.sum(true_positives) / ivy.sum(all_positives)
    elif average == "macro":
        precision = ivy.mean(true_positives / all_positives)
    else:
        raise ValueError("Invalid value for 'average'.")
    return precision


""" Test for precision_score

import pytest
from sklearn.metrics import precision_score as sk_precision_score

# Import the precision_score function you provided
from your_module import precision_score

# Generate some example data
y_true = ivy.array([0, 1, 1, 0, 1, 0])
y_pred = ivy.array([0, 1, 0, 1, 0, 1])

def test_precision_score():
    # Calculate precision using above defined function
    ivy_precision = precision_score(y_true, y_pred)

    # Calculate precision using scikit-learn for comparison
    sklearn_precision = sk_precision_score(ivy.to_numpy(y_true), ivy.to_numpy(y_pred))

    # Check if both values are equal within a small tolerance
    assert ivy.to_numpy(ivy_precision) == pytest.approx(sklearn_precision, abs=1e-5)

# Run the test
if __name__ == "__main__":
    test_precision_score()
"""
