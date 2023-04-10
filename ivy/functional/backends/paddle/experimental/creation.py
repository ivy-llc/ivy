# global
import struct
from typing import Optional, Tuple
import math
import paddle
from ivy.utils.exceptions import IvyNotImplementedException
from paddle.fluid.libpaddle import Place
from ivy.functional.backends.paddle.device import to_device

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# noinspection PyProtectedMember
# Helpers for calculating Window Functions
# ----------------------------------------
# Code from cephes for i0

_i0A = [
    -4.41534164647933937950e-18,
    3.33079451882223809783e-17,
    -2.43127984654795469359e-16,
    1.71539128555513303061e-15,
    -1.16853328779934516808e-14,
    7.67618549860493561688e-14,
    -4.85644678311192946090e-13,
    2.95505266312963983461e-12,
    -1.72682629144155570723e-11,
    9.67580903537323691224e-11,
    -5.18979560163526290666e-10,
    2.65982372468238665035e-9,
    -1.30002500998624804212e-8,
    6.04699502254191894932e-8,
    -2.67079385394061173391e-7,
    1.11738753912010371815e-6,
    -4.41673835845875056359e-6,
    1.64484480707288970893e-5,
    -5.75419501008210370398e-5,
    1.88502885095841655729e-4,
    -5.76375574538582365885e-4,
    1.63947561694133579842e-3,
    -4.32430999505057594430e-3,
    1.05464603945949983183e-2,
    -2.37374148058994688156e-2,
    4.93052842396707084878e-2,
    -9.49010970480476444210e-2,
    1.71620901522208775349e-1,
    -3.04682672343198398683e-1,
    6.76795274409476084995e-1,
]

_i0B = [
    -7.23318048787475395456e-18,
    -4.83050448594418207126e-18,
    4.46562142029675999901e-17,
    3.46122286769746109310e-17,
    -2.82762398051658348494e-16,
    -3.42548561967721913462e-16,
    1.77256013305652638360e-15,
    3.81168066935262242075e-15,
    -9.55484669882830764870e-15,
    -4.15056934728722208663e-14,
    1.54008621752140982691e-14,
    3.85277838274214270114e-13,
    7.18012445138366623367e-13,
    -1.79417853150680611778e-12,
    -1.32158118404477131188e-11,
    -3.14991652796324136454e-11,
    1.18891471078464383424e-11,
    4.94060238822496958910e-10,
    3.39623202570838634515e-9,
    2.26666899049817806459e-8,
    2.04891858946906374183e-7,
    2.89137052083475648297e-6,
    6.88975834691682398426e-5,
    3.36911647825569408990e-3,
    8.04490411014108831608e-1,
]


def _i0(x):
    # Modified Bessel function of the first kind, order 0.

    def _i0_1(x):
        return paddle.exp(x) * _chbevl(x / 2.0 - 2, _i0A)

    def _i0_2(x):
        return paddle.exp(x) * _chbevl(32.0 / x - 2.0, _i0B) / paddle.sqrt(x)

    def _chbevl(x, vals):
        b0 = vals[0]
        b1 = 0.0

        for i in range(1, len(vals)):
            b2 = b1
            b1 = b0
            b0 = x * b1 - b2 + vals[i]

        return 0.5 * (b0 - b2)

    x = paddle.to_tensor(x)
    if "complex" in str(x.dtype):
        raise ivy.exceptions.IvyException("i0 not supported for complex values")
    if "float" in str(x.dtype):
        x = x.cast(ivy.default_float_dtype())
    x = paddle.abs(x)
    return paddle.where(x <= 8.0, _i0_1(x), _i0_2(x))


def _kaiser_window(window_length, beta):
    if window_length == 1:
        result_dtype = ivy.promote_types_of_inputs(window_length, 0.0)[0].dtype
        return paddle.ones(1, dtype=result_dtype)
    n = paddle.arange(0, window_length)
    alpha = (window_length - 1) / 2.0
    return _i0(beta * paddle.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) / _i0(float(beta))


# Array API Standard #
# -------------------#


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    /,
    *,
    device: Place,
) -> Tuple[paddle.Tensor]:
    return to_device(
        paddle.triu_indices(n_rows, col=n_cols, offset=k, dtype="int64"), device
    )


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if periodic is False:
        return _kaiser_window(window_length, beta).cast(dtype)
    else:
        return _kaiser_window(window_length + 1, beta)[:-1].cast(dtype)


def hamming_window(
    window_length: int,
    /,
    *,
    periodic: Optional[bool] = True,
    alpha: Optional[float] = 0.54,
    beta: Optional[float] = 0.46,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    # Implemented as a composite function in ivy.functional.experimental.creation
    raise IvyNotImplementedException()


def vorbis_window(
    window_length: paddle.Tensor,
    *,
    dtype: Optional[paddle.dtype] = paddle.float32,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    i = paddle.arange(1, window_length * 2, 2)
    pi = paddle.to_tensor(math.pi)
    return paddle.sin((pi / 2) * (paddle.sin(pi * i / (window_length * 2)) ** 2)).cast(
        dtype
    )


def hann_window(
    size: int,
    /,
    *,
    periodic: Optional[bool] = True,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    size = size + 1 if periodic else size
    pi = paddle.to_tensor(math.pi)
    result_dtype = ivy.promote_types_of_inputs(size, 0.0)[0].dtype
    if size < 1:
        return paddle.to_tensor([], dtype=result_dtype)
    if size == 1:
        return paddle.ones(1, dtype=result_dtype)
    n = paddle.arange(1 - size, size, 2)
    res = (0.5 + 0.5 * paddle.cos(pi * n / (size - 1))).cast(dtype)
    return res[:-1] if periodic else res


def tril_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    /,
    *,
    device: Place,
) -> Tuple[paddle.Tensor, ...]:
    return tuple(
        to_device(
            paddle.tril_indices(n_rows, col=n_cols, offset=k, dtype="int64"), device
        )
    )


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "bfloat16",
            "complex64",
            "complex128",
            "uint16",
            "uint32",
            "uint64",
        )
    },
    backend_version,
)
def frombuffer(
    buffer: bytes,
    dtype: Optional[paddle.dtype] = float,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> paddle.Tensor:
    dtype_bytes = int(ivy.Dtype(dtype).dtype_bits / 8)
    if str(dtype) == "bool":
        dtype_bytes = 1
    dtype_str = str(dtype)
    struct_format = {
        "bool": "?",
        "int8": "b",
        "int16": "h",
        "int32": "i",
        "int64": "q",
        "uint8": "B",
        "float16": "e",
        "float32": "f",
        "float64": "d",
    }
    ret = []
    for i in range(0, len(buffer), dtype_bytes):
        x = struct.unpack(struct_format[dtype_str], buffer[i : i + dtype_bytes])
        ret = ret + list(x)
    if offset > 0:
        offset = int(offset / dtype_bytes)
    if count > -1:
        ret = ret[offset : offset + count]
    else:
        ret = ret[offset:]
    ret = paddle.to_tensor(ret, dtype=dtype)

    return ret
