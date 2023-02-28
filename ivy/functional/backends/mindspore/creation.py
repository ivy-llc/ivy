# global

from numbers import Number
from typing import Union, List, Optional, Sequence
import numpy as orig_np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Type
from mindspore.ops import functional as F


# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_unsupported_device_and_dtypes,
    _get_first_array,

)
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
)
from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException


# Array API Standard #
# -------------------#



@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
def asarray(
    obj: Union[
        ms.Tensor, bool, int, float, NestedSequence, SupportsBufferProtocol
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Type = None,
    device: str,
    out: ms.Tensor = None,
) -> ms.Tensor:
    if copy:
        if dtype is None and isinstance(obj, ms.Tensor):
            return ops.identity(obj)
        if dtype is None and not isinstance(obj, ms.Tensor):
            try:
                dtype = ivy.default_dtype(item=obj, as_native=True)
                tensor = ms.Tensor(obj, dtype=dtype)
            except (TypeError, ValueError):
                dtype = ivy.default_dtype(dtype=dtype, item=obj, as_native=True)
                tensor = ms.Tensor(
                    ivy.nested_map(obj, lambda x: F.cast(x, dtype)),
                    dtype=dtype,
                )
            return ops.identity(F.cast(tensor, dtype))
        else:
            dtype = ivy.as_ivy_dtype(ivy.default_dtype(dtype=dtype, item=obj))
            try:
                tensor = ms.Tensor(obj, dtype=dtype)
            except (TypeError, ValueError):
                tensor = ms.Tensor(
                    ivy.nested_map(obj, lambda x: F.cast(x, dtype)),
                    dtype=dtype,
                )
            return ops.identity(F.cast(tensor, dtype))
    else:
        if dtype is None and isinstance(obj, ms.Tensor):
            return obj
        if dtype is None and not isinstance(obj, ms.Tensor):
            try:
                dtype = ivy.default_dtype(item=obj, as_native=True)
                return ms.Tensor(obj, dtype=dtype)
            except (TypeError, ValueError):
                dtype = ivy.as_ivy_dtype(ivy.default_dtype(dtype=dtype, item=obj))
                return ms.Tensor(
                    ivy.nested_map(obj, lambda x: F.cast(x, dtype)),
                    dtype=dtype,
                )
        else:
            dtype = ivy.default_dtype(dtype=dtype, item=obj, as_native=True)
            try:
                tensor = ms.Tensor(obj, dtype=dtype)
            except (TypeError, ValueError):
                print('no this ValueError', )
                tensor = ms.Tensor(
                    ivy.nested_map(obj, lambda x: F.cast(x, dtype)),
                    dtype=dtype,
                )
            return F.cast(tensor, dtype)
