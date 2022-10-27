from typing import Optional, Union
import numpy as np
from ivy.functional.backends.numpy.helpers import _handle_0_dim_output


@_handle_0_dim_output
def sinc(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sinc(x).astype(x.dtype)


@_handle_0_dim_output
def lcm(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.abs(
        np.lcm(
            x1,
            x2,
            out=out,
        )
    )


lcm.support_native_out = True


@_handle_0_dim_output
def fmod(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.fmod(
        x1,
        x2,
        out=None,
    )


fmod.support_native_out = True


@_handle_0_dim_output
def fmax(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
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


def trapz(
    y: np.ndarray,
    /,
    *,
    x: Optional[np.ndarray] = None,
    dx: Optional[float] = 1.0,
    axis: Optional[int] = -1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.trapz(y, x=x, dx=dx, axis=axis)


trapz.support_native_out = False


def float_power(
    x1: Union[np.ndarray, float, list, tuple],
    x2: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.asarray(np.float_power(x1, x2, out=out), dtype=x1.dtype)


float_power.support_native_out = True


def exp2(
    x: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.exp2(x, out=out)


exp2.support_native_out = True
