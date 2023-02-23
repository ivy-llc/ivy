# global
import numpy as np
from typing import Optional, Sequence, Union
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


def percentile(a,
               q,
               /,
               *,
               axis=None,
               out=None,
               overwrite_input=False,
               method="linear",
               keepdims=False,
               interpolation=None):

    axis = tuple(axis) if isinstance(axis, list) else axis
    a = ivy.astype(ivy.array(a))
    ret = ivy.percentile(a, q, axis=axis, overwrite_input=overwrite_input, method=method, keepdims=keepdims,
                         interpolation=interpolation, out=out)

    return ret
