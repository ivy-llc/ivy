import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from sklearn.utils.multiclass import type_of_target
from ivy.utils.exceptions import IvyValueError


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
def recall_score(y_true, y_pred, *, average="binary", sample_weight=None):
    # Determine the type of the target variable
    y_type = type_of_target(y_true)
    # Check if the 'average' parameter has a valid value
    if average != "binary" and average != "micro" and average != "macro":
        raise IvyValueError(
            "Invalid value for 'average'. Supported values are 'binary', 'micro', or"
            " 'macro'."
        )
    # Calculate true positive and actual positive counts based on the target type
    if y_type.startswith("multilabel"):
        true_positive = ivy.sum(ivy.logical_and(y_true, y_pred) * sample_weight, axis=1)
        actual_positive = ivy.sum(y_true * sample_weight, axis=1)
    else:
        true_positive = ivy.sum(ivy.logical_and(y_true, y_pred) * sample_weight)
        actual_positive = ivy.sum(y_true * sample_weight)
    # Calculate recall for each class or overall
    recall = true_positive / ivy.maximum(
        actual_positive, ivy.to_scalar(ivy.array([1], "float64"))
    )
    # Perform additional calculations for micro or macro averaging
    if average == "micro":
        recall = ivy.sum(true_positive) / ivy.maximum(
            ivy.sum(actual_positive), ivy.to_scalar(ivy.array([1], "float64"))
        )
    elif average == "macro":
        recall = ivy.mean(recall)
    return recall
