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
def f1_score(y_true, y_pred, *, sample_weight=None):
    # Ensure that y_true and y_pred have the same shape
    if y_true.shape != y_pred.shape:
        raise IvyValueError("y_true and y_pred must have the same shape")

    # Check if sample_weight is provided and normalize it
    if sample_weight is not None:
        sample_weight = ivy.array(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise IvyValueError(
                "sample_weight must have the same length as y_true and y_pred"
            )
        sample_weight = sample_weight / ivy.sum(sample_weight)
    else:
        sample_weight = ivy.ones_like(y_true)

    # Calculate true positives, predicted positives, and actual positives
    true_positives = ivy.logical_and(ivy.equal(y_true, 1), ivy.equal(y_pred, 1)).astype(
        "int64"
    )
    predicted_positives = ivy.equal(y_pred, 1).astype("int64")
    actual_positives = ivy.equal(y_true, 1).astype("int64")

    # Apply sample weights
    weighted_true_positives = ivy.multiply(true_positives, sample_weight)
    weighted_predicted_positives = ivy.multiply(predicted_positives, sample_weight)
    weighted_actual_positives = ivy.multiply(actual_positives, sample_weight)

    # Compute precision and recall with checks for division by zero
    precision = 0.0
    recall = 0.0
    if ivy.sum(weighted_predicted_positives) > 0:
        precision = ivy.sum(weighted_true_positives) / ivy.sum(
            weighted_predicted_positives
        )
    if ivy.sum(weighted_actual_positives) > 0:
        recall = ivy.sum(weighted_true_positives) / ivy.sum(weighted_actual_positives)

    # Compute F1 score with a check to avoid division by zero
    if precision + recall > 0:
        ret = 2 * (precision * recall) / (precision + recall)
    else:
        ret = ivy.array(0.0)  # If both precision and recall are zero, F1 score is zero

    ret = ret.astype("float64")
    return ret


@to_ivy_arrays_and_back
def precision_score(y_true, y_pred, *, sample_weight=None):
    # Ensure that y_true and y_pred have the same shape
    if y_true.shape != y_pred.shape:
        raise IvyValueError("y_true and y_pred must have the same shape")

    # Check if sample_weight is provided and normalize it
    if sample_weight is not None:
        sample_weight = ivy.array(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise IvyValueError(
                "sample_weight must have the same length as y_true and y_pred"
            )
        sample_weight = sample_weight / ivy.sum(sample_weight)
    else:
        sample_weight = ivy.ones_like(y_true)
    # Calculate true positives and predicted positives
    true_positives = ivy.logical_and(ivy.equal(y_true, 1), ivy.equal(y_pred, 1)).astype(
        "int64"
    )
    predicted_positives = ivy.equal(y_pred, 1).astype("int64")

    # Apply sample weights
    weighted_true_positives = ivy.multiply(true_positives, sample_weight)
    weighted_predicted_positives = ivy.multiply(predicted_positives, sample_weight)

    # Compute precision
    ret = ivy.sum(weighted_true_positives) / ivy.sum(weighted_predicted_positives)

    ret = ret.astype("float64")
    return ret


@to_ivy_arrays_and_back
def recall_score(y_true, y_pred, *, sample_weight=None):
    # Ensure that y_true and y_pred have the same shape
    if y_true.shape != y_pred.shape:
        raise IvyValueError("y_true and y_pred must have the same shape")

    # Check if sample_weight is provided and normalize it
    if sample_weight is not None:
        sample_weight = ivy.array(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise IvyValueError(
                "sample_weight must have the same length as y_true and y_pred"
            )
        sample_weight = sample_weight / ivy.sum(sample_weight)
    else:
        sample_weight = ivy.ones_like(y_true)
    # Calculate true positives and actual positives
    true_positives = ivy.logical_and(ivy.equal(y_true, 1), ivy.equal(y_pred, 1)).astype(
        "int64"
    )
    actual_positives = ivy.equal(y_true, 1).astype("int64")

    # Apply sample weights
    weighted_true_positives = ivy.multiply(true_positives, sample_weight)
    weighted_actual_positives = ivy.multiply(actual_positives, sample_weight)

    # Compute recall
    ret = ivy.sum(weighted_true_positives) / ivy.sum(weighted_actual_positives)

    ret = ret.astype("float64")
    return ret
