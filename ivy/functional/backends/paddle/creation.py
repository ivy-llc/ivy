# global

from numbers import Number
from typing import Union, List, Optional, Sequence

import numpy as np
import paddle

# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_unsupported_device_and_dtypes,
    _get_first_array,
)
from ivy.functional.ivy.creation import (
    NestedSequence,
    SupportsBufferProtocol,
)
from . import backend_version
from ivy.exceptions import IvyNotImplementedException
from paddle.fluid.libpaddle import Place


# Array API Standard #
# -------------------#


def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def asarray(
    obj: Union[
        paddle.Tensor,
        np.ndarray,
        bool,
        int,
        float,
        NestedSequence,
        SupportsBufferProtocol,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    raise IvyNotImplementedException()


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def empty_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def from_dlpack(x, /, *, out: Optional[paddle.Tensor] = None):
    raise IvyNotImplementedException()


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


full.support_native_out = True


def full_like(
    x: paddle.Tensor,
    /,
    fill_value: Number,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def linspace(
    start: Union[paddle.Tensor, float],
    stop: Union[paddle.Tensor, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def meshgrid(
    *arrays: paddle.Tensor,
    sparse: bool = False,
    indexing: str = "xy",
) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def ones_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def tril(
    x: paddle.Tensor, /, *, k: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def triu(
    x: paddle.Tensor, /, *, k: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def zeros_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


# Extra #
# ------#


array = asarray


def copy_array(
    x: paddle.Tensor,
    *,
    to_ivy_array: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def one_hot(
    indices: paddle.Tensor,
    depth: int,
    /,
    *,
    on_value: Optional[paddle.Tensor] = None,
    off_value: Optional[paddle.Tensor] = None,
    axis: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
