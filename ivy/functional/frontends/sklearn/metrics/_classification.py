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
def recall_score(y_true, y_pred, *, sample_weight=None):
    # TODO: implement sample_weight
    y_type = type_of_target(y_true)
    if y_type.startswith("multilabel"):
        raise ValueError("Multilabel not supported for recall score")
    else:
        true_positives = ivy.logical_and(ivy.equal(y_true, 1), ivy.equal(y_pred, 1)).astype("int64")
        actual_positives = ivy.equal(y_true, 1).astype("int64")
        ret = ivy.sum(true_positives).astype("int64")
        actual_pos_count = ivy.sum(actual_positives).astype("int64")
        ret = ret / actual_pos_count
        ret = ret.astype("float64")
    return ret
