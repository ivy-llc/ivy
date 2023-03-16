from numbers import Number
from typing import Optional, Tuple, Union

import paddle


import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Array API Standard #
# ------------------ #


def argmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    select_last_index: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def argmin(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[paddle.dtype] = None,
    select_last_index: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nonzero(
    x: paddle.Tensor,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[paddle.Tensor, Tuple[paddle.Tensor]]:
    raise IvyNotImplementedException()


@with_unsupported_dtypes(
    {"2.4.2 and below": ("uint16", "bfloat16")},
    backend_version,
)
def where(
    condition: paddle.Tensor,
    x1: Union[float, int, paddle.Tensor],
    x2: Union[float, int, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    x1, x2 = ivy.broadcast_arrays(x1, x2)
    ret_dtype = x1.dtype
    arrays = [condition, x1, x2]
    scalar_out = all(map(lambda x: x.ndim == 0, arrays))
    for i, array in enumerate(arrays):
        if array.ndim == 0:
            arrays[i] = ivy.to_native(ivy.expand_dims(array, axis=0))
        else:
            arrays[i] = ivy.to_native(arrays[i])
    condition, x1, x2 = arrays

    if ret_dtype in ['int8', 'int16', 'uint8', 'uint16', 'bfloat16', 'float16', 'bool']:
        x1 = x1.cast('float32')
        x2 = x2.cast("float32")
        result = paddle.where(condition, x1, x2)
    elif ret_dtype in ['complex64', 'complex128']:
        result_real = paddle.where(condition, paddle.real(x1), paddle.real(x2))
        result_imag = paddle.where(condition, paddle.imag(x1), paddle.imag(x2))
        result = result_real + 1j * result_imag
    else:
        result = paddle.where(condition, x1, x2)

    return result.squeeze().cast(ret_dtype) if scalar_out else result.cast(ret_dtype)


# Extra #
# ----- #


def argwhere(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    raise IvyNotImplementedException()
