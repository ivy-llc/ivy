# global
from typing import Any, Sequence

import ivy


# squeeze
def squeeze(
    x: Sequence[Any],
    axis: int,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
    signature=None,
    extobj=None,
) -> Any:

    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.squeeze(x, axis, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


squeeze.unsupported_dtypes = {"torch": ("float16",)}
