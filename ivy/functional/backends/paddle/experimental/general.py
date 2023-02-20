from typing import Optional
import paddle
from ivy.exceptions import IvyNotImplementedException

from . import backend_version


def isin(
    elements: paddle.Tensor,
    test_elements: paddle.Tensor,
    /,
    *,
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
