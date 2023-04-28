from typing import Union, Optional, Tuple, List
from numbers import Number


def sinc(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.sinc Not Implemented")


def lcm(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.lcm Not Implemented")


def fmax(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.fmax Not Implemented")


def fmin(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.fmin Not Implemented")


def trapz(
    y: Union[(None, tf.Variable)],
    /,
    *,
    x: Optional[Union[(None, tf.Variable)]] = None,
    dx: float = 1.0,
    axis: int = (-1),
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.trapz Not Implemented")


def float_power(
    x1: Union[(None, tf.Variable, float, list, tuple)],
    x2: Union[(None, tf.Variable, float, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.float_power Not Implemented")


def exp2(
    x: Union[(None, tf.Variable, float, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.exp2 Not Implemented")


def copysign(
    x1: Union[(None, tf.Variable, Number)],
    x2: Union[(None, tf.Variable, Number)],
    /,
    *,
    out: Optional[None] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.copysign Not Implemented")


def count_nonzero(
    a: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(int, Tuple[(int, ...)])]] = None,
    keepdims: bool = False,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.count_nonzero Not Implemented")


def nansum(
    x: Union[(None, tf.Variable)],
    /,
    *,
    axis: Optional[Union[(Tuple[(int, ...)], int)]] = None,
    dtype: Optional[None] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.nansum Not Implemented")


def gcd(
    x1: Union[(None, tf.Variable, int, list, tuple)],
    x2: Union[(None, tf.Variable, float, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.gcd Not Implemented")


def isclose(
    a: Union[(None, tf.Variable)],
    b: Union[(None, tf.Variable)],
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.isclose Not Implemented")


def nan_to_num(
    x: Union[(None, tf.Variable)],
    /,
    *,
    copy: bool = True,
    nan: Union[(float, int)] = 0.0,
    posinf: Optional[Union[(float, int)]] = None,
    neginf: Optional[Union[(float, int)]] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.nan_to_num Not Implemented")


def logaddexp2(
    x1: Union[(None, tf.Variable, float, list, tuple)],
    x2: Union[(None, tf.Variable, float, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.logaddexp2 Not Implemented")


def signbit(
    x: Union[(None, tf.Variable, float, int, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.signbit Not Implemented")


def hypot(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.hypot Not Implemented")


def allclose(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> bool:
    raise NotImplementedError("mxnet.allclose Not Implemented")


def fix(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.fix Not Implemented")


def nextafter(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.nextafter Not Implemented")


def diff(
    x: Union[(None, tf.Variable, list, tuple)],
    /,
    *,
    n: int = 1,
    axis: int = (-1),
    prepend: Optional[Union[(None, tf.Variable, int, float, list, tuple)]] = None,
    append: Optional[Union[(None, tf.Variable, int, float, list, tuple)]] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.diff Not Implemented")


def angle(
    input: Union[(None, tf.Variable)],
    /,
    *,
    deg: Optional[bool] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.angle Not Implemented")


def imag(
    val: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.imag Not Implemented")


def zeta(
    x: Union[(None, tf.Variable)],
    q: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
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
    x: Union[(None, tf.Variable)],
    y: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.xlogy Not Implemented")


def real(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.real Not Implemented")


def conj(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.conj Not Implemented")


def ldexp(
    x1: Union[(None, tf.Variable)],
    x2: Union[(None, tf.Variable, int)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.ldexp Not Implemented")


def frexp(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[
        Union[(Tuple[(None, None)], Tuple[(tf.Variable, tf.Variable)])]
    ] = None,
) -> Union[(Tuple[(None, None)], Tuple[(tf.Variable, tf.Variable)])]:
    raise NotImplementedError("mxnet.frexp Not Implemented")
