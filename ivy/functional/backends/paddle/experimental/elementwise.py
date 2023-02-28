# global
from typing import Optional, Union, Tuple, List
from numbers import Number
from math import pi
import paddle
from ivy.utils.exceptions import IvyNotImplementedException

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
    return paddle.lcm(x1, x2)


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
    return paddle.fmax(x1, x2)


def fmin(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.fmin(x1, x2)


def sinc(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.where(
        x == 0,
        1,
        paddle.divide(paddle.sin(x), x)
    )


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
    if out is not None:
        raise ValueError("The 'out' argument is not supported in this implementation.")
    return paddle.nansum(x, axis=axis, dtype=dtype)


def gcd(
    x1: Union[paddle.Tensor, int, list, tuple],
    x2: Union[paddle.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.gcd(x1, x2)


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
    if out is not None:
        raise ValueError("The 'out' argument is not supported in this implementation.")
    return paddle.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def angle(
    input_tensor: paddle.Tensor,
    /,
    *,
    deg: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
    name: Optional[str] = None,
) -> paddle.Tensor:
    if deg is not None:
        raise ValueError("The 'deg' argument is not supported in this implementation.")
    if out is not None:
        return paddle.angle(input_tensor, out=out, name=name)
    else:
        return paddle.angle(input_tensor, name=name)


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
    if copy:
        x = x.clone()
    nan_mask = paddle.isnan(x)
    if nan_mask.any():
        x[nan_mask] = nan
    if posinf is not None:
        posinf_mask = paddle.isinf(x) & (x > 0)
        if posinf_mask.any():
            x[posinf_mask] = posinf
    if neginf is not None:
        neginf_mask = paddle.isinf(x) & (x < 0)
        if neginf_mask.any():
            x[neginf_mask] = neginf
    return x



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
    x=paddle.to_tensor(x)
    if out is not None:
        raise ValueError("The 'out' argument is not supported in this implementation.")
    return paddle.diff(x, n=n, axis=axis, prepend=prepend, append=append)


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