# global
import paddle
from typing import Tuple, Optional
from collections import namedtuple

# local

from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException


def unique_all(
    x: paddle.Tensor,
    /,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        # Flatten the tensor to 1D
        flat_x = paddle.flatten(x)
        sorted_x = paddle.sort(flat_x)[0]
        indices = paddle.where(sorted_x[1:] != sorted_x[:-1])[0] + 1
        indices = paddle.concat([paddle.to_tensor([0]), indices, paddle.to_tensor([len(flat_x)])])
        if len(indices) == 2:
            unique_x = sorted_x[indices[:-1]]
            counts = paddle.to_tensor([len(flat_x)])
        else:
            unique_x = sorted_x[indices[:-1]]
            counts = indices[1:] - indices[:-1]

        return unique_x


def unique_counts(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_inverse(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_values(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
