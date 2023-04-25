# global
from typing import Optional, Union, Tuple, List
from numbers import Number
import paddle
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
    with_unsupported_device_and_dtypes,
)

# local
import ivy
from ivy import promote_types_of_inputs
from .. import backend_version


def lcm(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1_dtype = x1.dtype
    x2_dtype = x2.dtype
    if (x1_dtype, x2_dtype) == (paddle.int16, paddle.int16):
        return paddle.cast(
            paddle.lcm(paddle.cast(x1, paddle.int32), paddle.cast(x2, paddle.int32)),
            paddle.int16,
        )
    elif x1_dtype != x2_dtype:
        x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.lcm(x1, x2)


@with_supported_dtypes(
    {"2.4.2 and below": ("float64", "float32", "int64", "int64")},
    backend_version,
)
def fmax(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x1.dtype != x2.dtype:
        x1, x2 = promote_types_of_inputs(x1, x2)
    return paddle.fmax(x1, x2)


@with_supported_dtypes(
    {"2.4.2 and below": ("float64", "float32", "int64", "int64")},
    backend_version,
)
def fmin(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x1.dtype != x2.dtype:
        x1, x2 = promote_types_of_inputs(x1, x2)
    return paddle.fmin(x1, x2)


def sinc(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.where(x == 0, 1, paddle.divide(paddle.sin(x), x))


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


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def float_power(
    x1: Union[paddle.Tensor, float, list, tuple],
    x2: Union[paddle.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1 = paddle.cast(x1, dtype="float64")
    x2 = paddle.cast(x2, dtype="float64")  # Compute the element-wise power
    return paddle.cast(paddle.pow(x1, x2), dtype=paddle.float64)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def exp2(
    x: Union[paddle.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        return ivy.pow(2, x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def copysign(
    x1: Union[paddle.Tensor, Number],
    x2: Union[paddle.Tensor, Number],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        x2 = ivy.where(ivy.equal(x2, paddle.to_tensor(0)), ivy.divide(1, x2), x2)
        signs = ivy.sign(x2)
        return ivy.multiply(ivy.abs(x1), signs)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint8", "int8", "int16", "float16")}}, backend_version
)
def nansum(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int, ...], int]] = None,
    dtype: Optional[paddle.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.nansum(x, axis=axis, dtype=dtype, keepdim=keepdims)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("int8", "int16")}}, backend_version
)
def gcd(
    x1: Union[paddle.Tensor, int, list, tuple],
    x2: Union[paddle.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return paddle.gcd(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("float16",)}}, backend_version
)
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
    return paddle.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def angle(
    input: paddle.Tensor,
    /,
    *,
    deg: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    result = paddle.angle(input)
    if deg:
        result = paddle.rad2deg(result)
    return result


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "bool",
        )
    },
    backend_version,
)
def imag(
    val: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.imag(val)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("uint16", "bfloat16")},
    backend_version,
)
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
    with ivy.ArrayMode(False):
        if ivy.is_int_dtype(x):
            if posinf is None:
                posinf = ivy.iinfo(x).max
            if neginf is None:
                neginf = ivy.iinfo(x).min
        elif ivy.is_float_dtype(x) or ivy.is_complex_dtype(x):
            if posinf is None:
                posinf = ivy.finfo(x).max
            if neginf is None:
                neginf = ivy.finfo(x).min
        ret = ivy.where(ivy.isnan(x), paddle.to_tensor(nan, dtype=x.dtype), x)
        ret = ivy.where(
            ivy.logical_and(ivy.isinf(ret), ret > 0),
            paddle.to_tensor(posinf, dtype=x.dtype),
            ret,
        )
        ret = ivy.where(
            ivy.logical_and(ivy.isinf(ret), ret < 0),
            paddle.to_tensor(neginf, dtype=x.dtype),
            ret,
        )
        if copy:
            return ret.clone()
        else:
            x = ret
            return x


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "float16")}}, backend_version
)
def logaddexp2(
    x1: Union[paddle.Tensor, float, list, tuple],
    x2: Union[paddle.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        return ivy.log2(ivy.exp2(x1) + ivy.exp2(x2))


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
    x = paddle.to_tensor(x)
    return paddle.diff(x, n=n, axis=axis, prepend=prepend, append=append)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def signbit(
    x: Union[paddle.Tensor, float, int, list, tuple],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        return ivy.less_equal(x, 0)


def hypot(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
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
    return paddle.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


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


_BERNOULLI_COEFS = [
    12,
    -720,
    30240,
    -1209600,
    47900160,
    -1307674368000 / 691,
    74724249600,
    -10670622842880000 / 3617,
    5109094217170944000 / 43867,
    -802857662698291200000 / 174611,
    14101100039391805440000 / 77683,
    -1693824136731743669452800000 / 236364091,
    186134520519971831808000000 / 657931,
    -37893265687455865519472640000000 / 3392780147,
    759790291646040068357842010112000000 / 1723168255201,
    -134196726836183700385281186201600000000 / 7709321041217,
]


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "uint16",
                "bfloat16",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "float16",
                "bool",
            )
        }
    },
    backend_version,
)
def zeta(
    x: paddle.Tensor,
    q: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        s, a = ivy.promote_types_of_inputs(x, q)
        s_, a_ = paddle.unsqueeze(x, -1), paddle.unsqueeze(q, -1)
        N = M = (
            paddle.to_tensor(8.0, dtype="float32")
            if q.dtype == paddle.float32
            else paddle.to_tensor(8.0, dtype="float64")
        )
        assert M <= len(_BERNOULLI_COEFS)
        k = paddle.unsqueeze(ivy.arange(N, dtype=q.dtype), tuple(range(q.ndim)))
        S = paddle.sum((a_ + k) ** -s_, -1)
        Q = ivy.divide((q + N) ** (1 - x), x - 1)
        T0 = (q + N) ** -x
        m = paddle.unsqueeze(ivy.arange(2 * M, dtype=s.dtype), tuple(range(s.ndim)))
        s_over_a = (s_ + m) / (a_ + N)
        s_over_a = ivy.where(
            s_over_a == 0, paddle.ones_like(s_over_a) * 1e-20, s_over_a
        )
        T1 = paddle.cumprod(s_over_a, -1)[..., ::2]
        # t=np.array(T1)
        T1 = paddle.clip(T1, max=ivy.finfo(T1.dtype).max)
        coefs = paddle.unsqueeze(
            paddle.to_tensor(_BERNOULLI_COEFS[: T1.shape[-1]], dtype=T1.dtype),
            tuple(range(a.ndim)),
        )
        T1 = T1 / coefs
        T = T0 * (0.5 + paddle.sum(T1, -1))
        ans = S + Q + T
        mask = x < 1
        ans[mask] = ivy.nan
        return ans


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


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "bool",
        )
    },
    backend_version,
)
def real(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.real(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint8", "int8")}}, backend_version
)
def count_nonzero(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, list, tuple]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[paddle.dtype] = None,
    name: Optional[str] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    non_zero_count = paddle.sum(x != 0, axis=axis, keepdim=keepdims, name=name)
    return paddle.to_tensor(non_zero_count, dtype=dtype)


@with_supported_dtypes(
    {
        "2.4.2 and below": (
            "complex64",
            "complex128",
            "float32",
            "float64",
            "int32",
            "int64",
        )
    },
    backend_version,
)
def conj(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.conj(x)
