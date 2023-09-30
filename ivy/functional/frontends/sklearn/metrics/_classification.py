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
        raise IvyValueError("Invalid value for 'average'.")
    return precision
