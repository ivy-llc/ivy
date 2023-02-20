# global
from typing import Optional, Union, Tuple, List
from numbers import Number
from math import pi
import paddle
from ivy.exceptions import IvyNotImplementedException

# local
import ivy
from ivy import promote_types_of_inputs
from ivy.functional.backends.torch.elementwise import _cast_for_unary_op
from .. import backend_version


def lcm(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def fmod(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def fmax(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def fmin(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def sinc(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    raise IvyNotImplementedException()


def trapz(
    y: paddle.Tensor,
    /,
    *,
    x: Optional[paddle.Tensor] = None,
    dx: Optional[float] = None,
    axis: Optional[int] = -1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def float_power(
    x1: Union[paddle.Tensor, float, list, tuple],
    x2: Union[paddle.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def exp2(
    x: Union[paddle.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def copysign(
    x1: Union[paddle.Tensor, Number],
    x2: Union[paddle.Tensor, Number],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def count_nonzero(
    a: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nansum(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int, ...], int]] = None,
    dtype: Optional[paddle.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def gcd(
    x1: Union[paddle.Tensor, int, list, tuple],
    x2: Union[paddle.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def isclose(
    a: paddle.Tensor,
    b: paddle.Tensor,
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def angle(
    input: paddle.Tensor,
    /,
    *,
    deg: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def imag(
    val: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nan_to_num(
    x: paddle.Tensor,
    /,
    *,
    copy: Optional[bool] = True,
    nan: Optional[Union[float, int]] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def logaddexp2(
    x1: Union[paddle.Tensor, float, list, tuple],
    x2: Union[paddle.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def diff(
    x: Union[paddle.Tensor, list, tuple],
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[Union[paddle.Tensor, int, float, list, tuple]] = None,
    append: Optional[Union[paddle.Tensor, int, float, list, tuple]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def signbit(
    x: Union[paddle.Tensor, float, int, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def hypot(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def allclose(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> bool:
    raise IvyNotImplementedException()


def fix(
    x: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nextafter(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def zeta(
    x: paddle.Tensor,
    q: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def gradient(
    x: paddle.Tensor,
    /,
    *,
    spacing: Optional[Union[int, list, tuple]] = 1,
    axis: Optional[Union[int, list, tuple]] = None,
    edge_order: Optional[int] = 1,
) -> Union[paddle.Tensor, List[paddle.Tensor]]:
    raise IvyNotImplementedException()


def xlogy(
    x: paddle.Tensor, y: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def real(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    raise IvyNotImplementedException()
