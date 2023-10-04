import ivy
import numpy as np  # To handle the nans
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from sklearn.utils.multiclass import type_of_target
import warnings  # importing this to raise the warning for zero_division


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
def precision_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn"
):
    """
    Calculate the precision score with support for sample weights and averaging options
    as seen in sklearn doc.

    Parameters:
    - y_true (ivy array): Ground truth target values.
    - y_pred (ivy array): Predicted target values.
    - labels (ivy array or None): List of labels to include (optional).
    - pos_label (int or str): The positive class label (default is 1).
    - average (str): Determines how precision is averaged.
    - sample_weight (ivy array or None): Sample weights (optional).
    - zero_division (str or float): Sets the value to return when there is a zero division.

    Returns:
    - Precision (array of floats or float): The calculated precision score(s).
    """
    # Convert PyTorch tensors to Ivy arrays
    y_true = ivy.array(y_true)
    y_pred = ivy.array(y_pred)

    # Calculate true positives, false positives, also with consideration for whether there is specified sample_weight or not
    true_positives = ivy.sum(
        ivy.where((y_true == pos_label) & (y_pred == pos_label), 1.0, 0.0)
    )
    false_positives = ivy.sum(
        ivy.where((y_true != pos_label) & (y_pred == pos_label), 1.0, 0.0)
    )

    if sample_weight is not None:
        sample_weight = ivy.array(sample_weight)
        weighted_true_positives = ivy.sum(
            sample_weight
            * ivy.where((y_true == pos_label) & (y_pred == pos_label), 1.0, 0.0)
        )
        weighted_false_positives = ivy.sum(
            sample_weight
            * ivy.where((y_true != pos_label) & (y_pred == pos_label), 1.0, 0.0)
        )
    else:
        weighted_true_positives = true_positives
        weighted_false_positives = false_positives

    # Handling zero division based on the provided zero_division parameter
    if zero_division == "warn":
        if weighted_true_positives + weighted_false_positives == 0:
            warnings.warn(
                "Precision is not well defined and being set to 0.0 due to zero"
                " division."
            )
            return ivy.array(0.0)
    elif zero_division == 0.0:
        if weighted_true_positives + weighted_false_positives == 0:
            return ivy.array(0.0)
    elif zero_division == 1.0:
        if weighted_true_positives + weighted_false_positives == 0:
            return ivy.array(1.0)
    elif zero_division == np.nan:
        if weighted_true_positives + weighted_false_positives == 0:
            return np.nan
    else:
        raise ValueError(
            "Invalid value for 'zero_division'. Use one of 'warn', 0.0, 1.0, or np.nan."
        )

    # Calculate precision for binary, micro and macro average values
    if average == "binary":
        return weighted_true_positives / (
            weighted_true_positives + weighted_false_positives
        )

    elif average == "micro":
        return weighted_true_positives / (
            weighted_true_positives + weighted_false_positives
        )

    elif average == "macro":
        label_precision = ivy.where(
            labels == pos_label,
            true_positives / (true_positives + false_positives),
            0.0,
        )
        return ivy.mean(label_precision)

    else:
        raise ValueError(
            "Invalid value for 'average'. Use one of 'binary', 'micro', or 'macro'."
        )
