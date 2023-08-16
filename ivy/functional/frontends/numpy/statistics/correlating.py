# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def corrcoef(x, y=None, /, *, rowvar=True, bias=None, ddof=None, dtype="float64"):
    if (bias is not None) or (ddof is not None):
        ivy.warn("bias and ddof are deprecated and have no effect")

    x = x.astype("float64")
    if y is not None:
        y = y.astype("float64")

    return ivy.corrcoef(x, y=y, rowvar=rowvar).astype(dtype)


@to_ivy_arrays_and_back
def correlate(a, v, mode=None, *, old_behavior=False):
    dtypes = [x.dtype for x in [a, v]]
    mode = mode if mode is not None else "valid"
    ivy.utils.assertions.check_equal(a.ndim, 1, as_array=False)
    ivy.utils.assertions.check_equal(v.ndim, 1, as_array=False)
    n = min(a.shape[0], v.shape[0])
    m = max(a.shape[0], v.shape[0])
    if a.shape[0] >= v.shape[0]:
        if mode == "full":
            r = n + m - 1
            for j in range(0, n - 1):
                a = ivy.concat((ivy.array([0]), a), axis=0)
        elif mode == "same":
            r = m
            right_pad = (n - 1) // 2
            left_pad = (n - 1) - (n - 1) // 2
            for _ in range(0, left_pad):
                a = ivy.concat((ivy.array([0]), a), axis=0)
            for _ in range(0, right_pad):
                a = ivy.concat((a, ivy.array([0])), axis=0)
        elif mode == "valid":
            r = m - n + 1
        else:
            raise ivy.utils.exceptions.IvyException("invalid mode")
        ret = ivy.array(
            [ivy.to_list((v[:n] * ivy.roll(a, -t)[:n]).sum()) for t in range(0, r)],
            dtype=max(dtypes),
        )
    else:
        if mode == "full":
            r = n + m - 1
            for j in range(0, n - 1):
                v = ivy.concat((ivy.array([0]), v), axis=0)
        elif mode == "same":
            r = m
            right_pad = (n - 1) // 2
            left_pad = (n - 1) - (n - 1) // 2
            for _ in range(0, left_pad):
                v = ivy.concat((ivy.array([0]), v), axis=0)
            for _ in range(0, right_pad):
                v = ivy.concat((v, ivy.array([0])), axis=0)
        elif mode == "valid":
            r = m - n + 1
        else:
            raise ivy.utils.exceptions.IvyException("invalid mode")
        ret = ivy.flip(
            ivy.array(
                [ivy.to_list((a[:n] * ivy.roll(v, -t)[:n]).sum()) for t in range(0, r)],
                dtype=max(dtypes),
            )
        )
    return ret
