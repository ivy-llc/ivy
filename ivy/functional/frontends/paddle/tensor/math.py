# local
# from ..math import *  # noqa: F401
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back

# NOTE:
# Only inplace functions are to be added in this file.
# Please add non-inplace counterparts to `/frontends/paddle/math.py`.


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def ceil_(x, name=None):
    return ivy.ceil(x, out=x)
