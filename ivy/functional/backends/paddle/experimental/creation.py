# global
from typing import Optional, Tuple
import math
import paddle
from ivy.utils.exceptions import IvyNotImplementedException
from paddle.fluid.libpaddle import Place
from ivy.functional.backends.paddle.device import to_device

# local
import ivy

# noinspection PyProtectedMember
# Helpers for calculating Window Functions
# ----------------------------------------
# Code from cephes for i0

_i0A = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
]

_i0B = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
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
    if 'complex' in str(x.dtype):
        raise ivy.exceptions.IvyException("i0 not supported for complex values")
    if 'float' in str(x.dtype):
        x = x.cast(ivy.default_float_dtype())
    x = paddle.abs(x)
    return paddle.where(x <= 8.0, _i0_1(x), _i0_2(x))


def _kaiser_window(window_length, beta):
    if window_length == 1:
        result_dtype = ivy.promote_types_of_inputs(window_length, 0.0)[0].dtype
        return paddle.ones(1, dtype=result_dtype)
    n = paddle.arange(0, window_length)
    alpha = (window_length - 1) / 2.0
    return _i0(beta * paddle.sqrt(1 - ((n - alpha) / alpha)**2.0)) / _i0(float(beta))


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
    raise IvyNotImplementedException()


def vorbis_window(
    window_length: paddle.Tensor,
    *,
    dtype: Optional[paddle.dtype] = paddle.float32,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def hann_window(
    size: int,
    /,
    *,
    periodic: Optional[bool] = True,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def tril_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    /,
    *,
    device: Place,
) -> Tuple[paddle.Tensor, ...]:

    return to_device(
        paddle.tril_indices(n_rows, col=n_cols, offset=k, dtype="int64"), device
    )
