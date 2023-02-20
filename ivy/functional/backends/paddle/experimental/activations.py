from typing import Optional, Union

# global
import paddle
from ivy.exceptions import IvyNotImplementedException


# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def logit(x: paddle.Tensor, /, *, eps: Optional[float] = None, out=None):
    raise IvyNotImplementedException()


def thresholded_relu(
    x: paddle.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
