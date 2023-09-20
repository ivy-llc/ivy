import ivy
from ivy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def check_classification_targets(y):
    """
    Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
        Target values.
    """
    y_type = type_of_target(y, input_name="y")
    if y_type not in [
        "binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences",
    ]:
        raise ValueError(
            f"Unknown label type: {y_type}. Maybe you are trying to fit a "
            "classifier, which expects discrete classes on a "
            "regression target with continuous values."
        )


# reapeated utility function
@to_ivy_arrays_and_back
def type_of_target(y, input_name="y"):
    # purely utility function
    unique_vals = len(ivy.unique_values(y))
    if y.ndim == 2 and y.shape[1] > 1 and unique_vals <= 2:
        return "multilabel-indicator"
    if y.ndim not in (1, 2):
        return "unknown"
    if ivy.is_float_dtype(y) and ivy.any(ivy.not_equal(y, y.astype("int64"))):
        return "continuous"
    else:
        if unique_vals > 2:
            return "multiclass"
        else:
            return "binary"
