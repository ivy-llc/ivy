# local
import ivy


def asanyarray(
    a,
    dtype=None,
    order=None,
    *,
    like=None
):
    return ivy.asanyarray(a, dtype=dtype, order=order, like=like)
