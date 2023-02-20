# global
import paddle
from ivy.exceptions import IvyNotImplementedException
from typing import Optional, Union


# msort
def msort(
    a: Union[paddle.Tensor, list, tuple], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


# lexsort
def lexsort(
    keys: paddle.Tensor, /, *, axis: int = -1, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
