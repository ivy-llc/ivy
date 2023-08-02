from typing import Optional, Union, Tuple, List
import numpy as np
import numpy.typing as npt

import ivy
from ivy import promote_types_of_inputs
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.25.2 and below": ("bfloat16",)}, backend_version)
def sinc(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sinc(x).astype(x.dtype)


@_scalar_output_to_0d_array
def fmax(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return np.fmax(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    )


fmax.support_native_out = True


@_scalar_output_to_0d_array
def float_power(
    x1: Union[np.ndarray, float, list, tuple],
    x2: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return np.float_power(x1, x2, out=out)


float_power.support_native_out = True


@_scalar_output_to_0d_array
def copysign(
    x1: npt.ArrayLike,
    x2: npt.ArrayLike,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not ivy.is_float_dtype(x1):
        x1 = x1.astype(ivy.default_float_dtype(as_native=True))
        x2 = x2.astype(ivy.default_float_dtype(as_native=True))
    return np.copysign(x1, x2, out=out)


copysign.support_native_out = True


@_scalar_output_to_0d_array
def count_nonzero(
    a: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    ret = np.count_nonzero(a, axis=axis, keepdims=keepdims)
    if np.isscalar(ret):
        return np.array(ret, dtype=dtype)
    return ret.astype(dtype)


count_nonzero.support_native_out = False


def nansum(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int, ...], int]] = None,
    dtype: Optional[np.dtype] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.nansum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


nansum.support_native_out = True


def isclose(
    a: np.ndarray,
    b: np.ndarray,
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if np.isscalar(ret):
        return np.array(ret, dtype="bool")
    return ret


isclose.support_native_out = False


def signbit(
    x: Union[np.ndarray, float, int, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.signbit(x, out=out)


signbit.support_native_out = True


def hypot(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.hypot(x1, x2)


def diff(
    x: Union[np.ndarray, list, tuple],
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[Union[np.ndarray, int, float, list, tuple]] = None,
    append: Optional[Union[np.ndarray, int, float, list, tuple]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    prepend = prepend if prepend is not None else np._NoValue
    append = append if append is not None else np._NoValue
    return np.diff(x, n=n, axis=axis, prepend=prepend, append=append)


diff.support_native_out = False


@_scalar_output_to_0d_array
def allclose(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[np.ndarray] = None,
) -> bool:
    return np.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


allclose.support_native_out = False


def fix(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.fix(x, out=out)


fix.support_native_out = True


def nextafter(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nextafter(x1, x2)


nextafter.support_natvie_out = True


def zeta(
    x: np.ndarray,
    q: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    temp = np.logical_and(np.greater(x, 0), np.equal(np.remainder(x, 2), 0))
    temp = np.logical_and(temp, np.less_equal(q, 0))
    temp = np.logical_and(temp, np.equal(np.remainder(q, 1), 0))
    inf_indices = np.logical_or(temp, np.equal(x, 1))
    temp = np.logical_and(np.not_equal(np.remainder(x, 2), 0), np.greater(x, 1))
    temp = np.logical_and(temp, np.less_equal(q, 0))
    nan_indices = np.logical_or(temp, np.less(x, 1))
    n, res = 1, 1 / q**x
    while n < 10000:
        term = 1 / (q + n) ** x
        n, res = n + 1, res + term
    ret = np.round(res, decimals=4)
    ret[nan_indices] = np.nan
    ret[inf_indices] = np.inf
    return ret


zeta.support_native_out = False


def gradient(
    x: np.ndarray,
    /,
    *,
    spacing: Union[int, list, tuple] = 1,
    axis: Optional[Union[int, list, tuple]] = None,
    edge_order: int = 1,
) -> Union[np.ndarray, List[np.ndarray]]:
    if type(spacing) in (int, float):
        return np.gradient(x, spacing, axis=axis, edge_order=edge_order)
    return np.gradient(x, *spacing, axis=axis, edge_order=edge_order)


def xlogy(
    x: np.ndarray, y: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x, y = promote_types_of_inputs(x, y)
    if (x == 0).all():
        return 0.0
    else:
        return x * np.log(y)


def conj(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.conj(x, out=out)
    if x.dtype == bool:
        return ret.astype("bool")
    return ret


def ldexp(
    x1: np.ndarray,
    x2: Union[np.ndarray, int, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.ldexp(x1, x2, out=out)


def frexp(
    x: np.ndarray, /, *, out: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if out is None:
        return np.frexp(x, out=(None, None))
    else:
        return np.frexp(x, out=out)


def modf(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.modf(x, out=out)


# --- LGAMMA --- #
LANCZOS_N = 13
lanczos_g = 6.024680040776729583740234375
lanczos_num_coeffs = np.array(
    [
        23531376880.410759688572007674451636754734846804940,
        42919803642.649098768957899047001988850926355848959,
        35711959237.355668049440185451547166705960488635843,
        17921034426.037209699919755754458931112671403265390,
        6039542586.3520280050642916443072979210699388420708,
        1439720407.3117216736632230727949123939715485786772,
        248874557.86205415651146038641322942321632125127801,
        31426415.585400194380614231628318205362874684987640,
        2876370.6289353724412254090516208496135991145378768,
        186056.26539522349504029498971604569928220784236328,
        8071.6720023658162106380029022722506138218516325024,
        210.82427775157934587250973392071336271166969580291,
        2.5066282746310002701649081771338373386264310793408,
    ]
)
lanczos_den_coeffs = np.array(
    [
        0.0,
        39916800.0,
        120543840.0,
        150917976.0,
        105258076.0,
        45995730.0,
        13339535.0,
        2637558.0,
        357423.0,
        32670.0,
        1925.0,
        66.0,
        1.0,
    ]
)


def sinpi(x):
    y = np.abs(x) % 2.0
    n = np.round(2.0 * y)
    assert 0 <= n and n <= 4

    if n == 0:
        r = np.sin(np.pi * y)
    elif n == 1:
        r = np.cos(np.pi * (y - 0.5))
    elif n == 2:
        r = np.sin(np.pi * (1.0 - y))
    elif n == 3:
        r = -np.cos(np.pi * (y - 1.5))
    elif n == 4:
        r = np.sin(np.pi * (y - 2.0))
    else:
        raise Exception("Unreachable code")

    return np.copysign(1.0, x) * r


def lanczos_sum(x):
    num = 0.0
    den = 0.0

    if x < 5.0:
        for i in range(LANCZOS_N - 1, -1, -1):
            num = num * x + lanczos_num_coeffs[i]
            den = den * x + lanczos_den_coeffs[i]
    else:
        for i in range(LANCZOS_N):
            num = num / x + lanczos_num_coeffs[i]
            den = den / x + lanczos_den_coeffs[i]

    return num / den


# TODO: Replace with native lgamma implementation when available
def lgamma(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    def func(x):
        if not np.isfinite(x):
            if np.isnan(x):
                return x  # lgamma(nan) = nan
            else:
                return np.inf  # lgamma(+-inf) = +inf

        if x == np.floor(x) and x <= 2.0:
            if x <= 0.0:
                return np.inf  # lgamma(n) = inf for integers n <= 0
            else:
                return 0.0  # lgamma(1) = lgamma(2) = 0.0

        absx = np.abs(x)
        if absx < 1e-20:
            return -np.log(absx)

        # Lanczos' formula
        r = np.log(lanczos_sum(absx)) - lanczos_g
        r += (absx - 0.5) * (np.log(absx + lanczos_g - 0.5) - 1)

        if x < 0.0:
            # Use reflection formula to get value for negative x.
            r = np.log(np.pi) - np.log(np.abs(sinpi(absx))) - np.log(absx) - r

        if np.isinf(r):
            raise OverflowError("Range error in lgamma")

        return r

    # Vectorize 'func' for element-wise operations on 'x', output matching 'x' dtype.
    vfunc = np.vectorize(func, otypes=[x.dtype])
    return vfunc(x)


# --- erfc --- #
# Polynomials for computing erf/erfc. Originally from cephes library.
# https://netlib.org/cephes/doubldoc.html
kErfcPCoefficient = np.array(
    [
        2.46196981473530512524e-10,
        5.64189564831068821977e-1,
        7.46321056442269912687e0,
        4.86371970985681366614e1,
        1.96520832956077098242e2,
        5.26445194995477358631e2,
        9.34528527171957607540e2,
        1.02755188689515710272e3,
        5.57535335369399327526e2,
    ]
)
kErfcQCoefficient = np.array(
    [
        1.00000000000000000000e0,
        1.32281951154744992508e1,
        8.67072140885989742329e1,
        3.54937778887819891062e2,
        9.75708501743205489753e2,
        1.82390916687909736289e3,
        2.24633760818710981792e3,
        1.65666309194161350182e3,
        5.57535340817727675546e2,
    ]
)
kErfcRCoefficient = np.array(
    [
        5.64189583547755073984e-1,
        1.27536670759978104416e0,
        5.01905042251180477414e0,
        6.16021097993053585195e0,
        7.40974269950448939160e0,
        2.97886665372100240670e0,
    ]
)
kErfcSCoefficient = np.array(
    [
        1.00000000000000000000e0,
        2.26052863220117276590e0,
        9.39603524938001434673e0,
        1.20489539808096656605e1,
        1.70814450747565897222e1,
        9.60896809063285878198e0,
        3.36907645100081516050e0,
    ]
)


# Evaluate the polynomial given coefficients and `x`.
# N.B. Coefficients should be supplied in decreasing order.
def _EvaluatePolynomial(x, coefficients):
    poly = np.full_like(x, 0.0)
    for c in coefficients:
        poly = poly * x + c
    return poly


# TODO: Remove this once native function is avilable.
# Compute an approximation of the error function complement (1 - erf(x)).
def erfc(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if x.dtype not in [np.float16, np.float32, np.float64]:
        raise ValueError("Input must be of type float16, float32, or float64.")

    input_dtype = x.dtype

    abs_x = np.abs(x)
    z = np.exp(-x * x)

    pp = _EvaluatePolynomial(abs_x, kErfcPCoefficient)
    pq = _EvaluatePolynomial(abs_x, kErfcQCoefficient)
    pr = _EvaluatePolynomial(abs_x, kErfcRCoefficient)
    ps = _EvaluatePolynomial(abs_x, kErfcSCoefficient)

    abs_x_small = abs_x < 8.0
    y = np.where(abs_x_small, z * pp / pq, z * pr / ps)
    result_no_underflow = np.where(x < 0.0, 2.0 - y, y)

    is_pos_inf = lambda op: np.logical_and(np.isinf(op), op > 0)
    underflow = np.logical_or(
        z == 0,
        np.logical_or(
            np.logical_and(is_pos_inf(pq), abs_x_small),
            np.logical_and(is_pos_inf(ps), np.logical_not(abs_x_small)),
        ),
    )
    result_underflow = np.where(x < 0, np.full_like(x, 2), np.full_like(x, 0))

    return np.where(underflow, result_underflow, result_no_underflow).astype(
        input_dtype
    )
