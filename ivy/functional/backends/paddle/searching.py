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
    {"2.4.2 and below": ("int8", "int16", "uint8", "uint16", "bfloat16", "float16", "complex64", "complex128", "bool")},
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
    if condition.rank().item()==0:
        condition= condition.unsqueeze(0)
        return paddle.where(condition, x1.data, x2.data).squeeze(0)
    
    return paddle.where(condition, x1.data, x2.data)
    


# Extra #
# ----- #


def argwhere(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    raise IvyNotImplementedException()
