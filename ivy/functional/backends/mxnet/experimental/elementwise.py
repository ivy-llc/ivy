from typing import Union, Optional, Tuple, List
from numbers import Number
import mxnet as mx


def sinc(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.sinc Not Implemented")


def lcm(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.lcm Not Implemented")


def fmax(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.fmax Not Implemented")


def fmin(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.fmin Not Implemented")


def trapz(
    y: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    x: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    dx: float = 1.0,
    axis: int = (-1),
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.trapz Not Implemented")


def float_power(
    x1: Union[(None, mx.ndarray.NDArray, float, list, tuple)],
    x2: Union[(None, mx.ndarray.NDArray, float, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.float_power Not Implemented")


def exp2(
    x: Union[(None, mx.ndarray.NDArray, float, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.exp2 Not Implemented")


def copysign(
    x1: Union[(None, mx.ndarray.NDArray, Number)],
    x2: Union[(None, mx.ndarray.NDArray, Number)],
    /,
    *,
    out: Optional[None] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.copysign Not Implemented")


def count_nonzero(
    a: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[(int, Tuple[(int, ...)])]] = None,
    keepdims: bool = False,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.count_nonzero Not Implemented")


def nansum(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[(Tuple[(int, ...)], int)]] = None,
    dtype: Optional[None] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.nansum Not Implemented")


def gcd(
    x1: Union[(None, mx.ndarray.NDArray, int, list, tuple)],
    x2: Union[(None, mx.ndarray.NDArray, float, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.gcd Not Implemented")


def isclose(
    a: Union[(None, mx.ndarray.NDArray)],
    b: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.isclose Not Implemented")


def nan_to_num(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: bool = True,
    nan: Union[(float, int)] = 0.0,
    posinf: Optional[Union[(float, int)]] = None,
    neginf: Optional[Union[(float, int)]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.nan_to_num Not Implemented")


def logaddexp2(
    x1: Union[(None, mx.ndarray.NDArray, float, list, tuple)],
    x2: Union[(None, mx.ndarray.NDArray, float, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.logaddexp2 Not Implemented")


def signbit(
    x: Union[(None, mx.ndarray.NDArray, float, int, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.signbit Not Implemented")


def hypot(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.hypot Not Implemented")


def allclose(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> bool:
    raise NotImplementedError("mxnet.allclose Not Implemented")


def fix(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.fix Not Implemented")


def nextafter(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.nextafter Not Implemented")


def diff(
    x: Union[(None, mx.ndarray.NDArray, list, tuple)],
    /,
    *,
    n: int = 1,
    axis: int = (-1),
    prepend: Optional[
        Union[(None, mx.ndarray.NDArray, int, float, list, tuple)]
    ] = None,
    append: Optional[Union[(None, mx.ndarray.NDArray, int, float, list, tuple)]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.diff Not Implemented")


def angle(
    input: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    deg: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.angle Not Implemented")


def imag(
    val: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.imag Not Implemented")


def zeta(
    x: Union[(None, mx.ndarray.NDArray)],
    q: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.zeta Not Implemented")


def gradient(
    x: None,
    /,
    *,
    spacing: Union[(int, list, tuple)] = 1,
    axis: Optional[Union[(int, list, tuple)]] = None,
    edge_order: int = 1,
) -> Union[(None, List[None])]:
    raise NotImplementedError("mxnet.gradient Not Implemented")


def xlogy(
    x: Union[(None, mx.ndarray.NDArray)],
    y: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.xlogy Not Implemented")


def real(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.real Not Implemented")


def conj(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.conj Not Implemented")


def ldexp(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray, int)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.ldexp Not Implemented")


def frexp(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[
        Union[(Tuple[(None, None)], Tuple[(mx.ndarray.NDArray, mx.ndarray.NDArray)])]
    ] = None,
) -> Union[(Tuple[(None, None)], Tuple[(mx.ndarray.NDArray, mx.ndarray.NDArray)])]:
    raise NotImplementedError("mxnet.frexp Not Implemented")
