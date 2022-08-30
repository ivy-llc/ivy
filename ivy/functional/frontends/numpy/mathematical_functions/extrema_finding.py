# global
import ivy
import numpy as np
from numpy import _NoValue

def minimum(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    if dtype:
        x1 = ivy.astype(ivy.array(x1), ivy.as_ivy_dtype(dtype))
        x2 = ivy.astype(ivy.array(x2), ivy.as_ivy_dtype(dtype))
    ret = ivy.minimum(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

#amin
def amin(
         a,
         axis=None,
         out=None,
         keepdims=_NoValue,
         initial=_NoValue,
         where=_NoValue
         ):
  ret=np.amin(a, axis=None, out=None, keepdims=_NoValue, initial=_NoValue,where=_NoValue)
  if ivy.is_array(where):
      ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
  return ret
