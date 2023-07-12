from typing import Optional, Union, Tuple, List
import numpy as np
import numpy.typing as npt

import ivy
from ivy import promote_types_of_inputs
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.25.1 and below": ("bfloat16",)}, backend_version)
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
    if x.dtype == np.bool:
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


# ---digamma---#
kLanczosGamma = 7  # aka g
kBaseLanczosCoeff = 0.99999999999980993227684700473478
kLanczosCoefficients = np.array(
    [
        676.520368121885098567009190444019,
        -1259.13921672240287047156078755283,
        771.3234287776530788486528258894,
        -176.61502916214059906584551354,
        12.507343278686904814458936853,
        -0.13857109526572011689554707,
        9.984369578019570859563e-6,
        1.50563273514931155834e-7,
    ]
)


def digamma(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Using `np.errstate` to ignore divide by zero error
    # to maintain the same behaviour as other frameworks.
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.asarray(x, dtype=x.dtype)
        zero = np.zeros_like(x)
        one_half = 0.5 * np.ones_like(x)
        one = np.ones_like(x)
        pi = np.pi * np.ones_like(x)
        lanczos_gamma = kLanczosGamma * np.ones_like(x)
        lanczos_gamma_plus_one_half = (kLanczosGamma + 0.5) * np.ones_like(x)
        log_lanczos_gamma_plus_one_half = np.log(kLanczosGamma + 0.5) * np.ones_like(x)
        base_lanczos_coeff = kBaseLanczosCoeff * np.ones_like(x)
        need_to_reflect = x < one_half
        z = np.where(need_to_reflect, -x, x - one)

        num = zero
        denom = base_lanczos_coeff
        for i in range(len(kLanczosCoefficients)):
            lanczos_coefficient = kLanczosCoefficients[i] * np.ones_like(x)
            index = i * np.ones_like(x)
            num = num - lanczos_coefficient / ((z + index + one) * (z + index + one))
            denom = denom + lanczos_coefficient / (z + index + one)

        t = lanczos_gamma_plus_one_half + z
        log_t = log_lanczos_gamma_plus_one_half + np.log1p(
            z / lanczos_gamma_plus_one_half
        )
        y = log_t + num / denom - lanczos_gamma / t

        reduced_x = x + np.abs(np.floor(x + 0.5))
        reflection = y - pi * np.cos(pi * reduced_x) / np.sin(pi * reduced_x)
        real_result = np.where(need_to_reflect, reflection, y)

        return np.where(
            np.logical_and(x <= zero, x == np.floor(x)), np.nan, real_result
        )
