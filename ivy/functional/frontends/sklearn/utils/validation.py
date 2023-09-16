import ivy
import numbers
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import with_unsupported_dtypes


# --- Helpers --- #
# --------------- #


@to_ivy_arrays_and_back
def _num_samples(x):
    message = "Expected sequence or array_like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = ivy.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        if isinstance(x.shape[0], numbers.Integral):
            ret = ivy.astype(x.shape[0], "int64")
            return ret

    try:
        ret = ivy.astype(len(x), "int64")
        return ret
    except TypeError as type_error:
        raise TypeError(message) from type_error


# --- Main --- #
# ------------ #


@to_ivy_arrays_and_back
def as_float_array(X, *, copy=True, force_all_finite=True):
    if X.dtype in [ivy.float32, ivy.float64]:
        return X.copy_array() if copy else X
    if ("bool" in X.dtype or "int" in X.dtype or "uint" in X.dtype) and ivy.itemsize(
        X
    ) <= 4:
        return_dtype = ivy.float32
    else:
        return_dtype = ivy.float64
    return ivy.asarray(X, dtype=return_dtype)


@to_ivy_arrays_and_back
def check_consistent_length(*arrays):
    lengths = [_num_samples(x) for x in arrays if x is not None]
    uniques = ivy.unique_values(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )


@with_unsupported_dtypes({"1.3.0 and below": ("complex",)}, "sklearn")
@to_ivy_arrays_and_back
def column_or_1d(y, *, warn=False):
    shape = y.shape
    if len(shape) == 2 and shape[1] == 1:
        y = ivy.reshape(y, (-1,))
    elif len(shape) > 2:
        raise ValueError("y should be a 1d array or a column vector")
    return y
