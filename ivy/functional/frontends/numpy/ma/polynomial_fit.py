import ivy

from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def roots(p):
    p = ivy.atleast_1d(p)
    if p.ndim != 1:
        raise ivy.utils.exceptions.IvyValueError("Input must be a rank-1 array.")

    non_zero = ivy.nonzero(p.flatten())[0]

    if len(non_zero) == 0:
        return ivy.array([])

    p = p[int(non_zero[0]) : int(non_zero[-1]) + 1]

    if (not ivy.is_complex_dtype(p)) or (not ivy.is_float_dtype(p)):
        p = p.astype(float)

    N = len(p)
    if N > 1:
        A = ivy.diag(
            ivy.ones((N - 2), dtype=p.dtype),
            k=-1,
        )
        A[0, :] = -p[1:] / p[0]
        # TODO: ivy.eigvals returns the wrong dtype in
        # some cases.
        roots = ivy.eigvals(A)
    else:
        roots = ivy.array([], dtype=ivy.float64)

    return roots
