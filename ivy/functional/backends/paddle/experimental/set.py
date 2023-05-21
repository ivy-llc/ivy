# global
from typing import Optional
import paddle
from ivy.utils.exceptions import IvyNotImplementedException

# local
import ivy
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from .. import backend_version


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def difference(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if isinstance(x1, list):
        x1 = paddle.concat(x1, axis=0)
    if isinstance(x2, list):
        x2 = paddle.concat(x2, axis=0)
    # remove duplicates from x1 and x2
    x1 = paddle.unique(x1)
    x2 = paddle.unique(x2)
    concatenated = paddle.concat([x1, x2], axis=0)
    unique, counts = paddle.unique(concatenated, return_counts=True)
    difference = unique[counts == 1]
    if out is not None:
        raise IvyNotImplementedException(
            "paddle backend does not currently support the 'out' kwarg for difference."
        )
    if difference.shape == [0]:
        return difference
    else:
        difference = difference.reshape([-1])
    if x1.dtype != paddle.bool:
        mask = ivy.isin(difference, x1)
        return difference[mask]
    else:
        # convert to numeric
        x1 = x1.astype(paddle.float32)
        difference = difference.astype(paddle.float32)
        mask = ivy.isin(difference, x1)
        return difference[mask].astype(paddle.bool)
