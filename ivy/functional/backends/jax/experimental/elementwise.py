import operator
from typing import Optional, Union, Tuple, List
from numbers import Number

from ivy import promote_types_of_inputs, default_float_dtype, is_float_dtype
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp
import jax.scipy as js

jax_ArrayLike = Union[JaxArray, Number]


def lcm(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.lcm(x1, x2)


def sinc(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinc(x)


def fmod(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.fmod(x1, x2)


def fmax(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.fmax(x1, x2)


def fmin(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fmin(x1, x2)


def trapz(
    y: JaxArray,
    /,
    *,
    x: Optional[JaxArray] = None,
    dx: Optional[float] = 1.0,
    axis: Optional[int] = -1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.trapz(y, x=x, dx=dx, axis=axis)


def float_power(
    x1: Union[JaxArray, float, list, tuple],
    x2: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.float_power(x1, x2)


def exp2(
    x: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.power(2, x)


def copysign(
    x1: jax_ArrayLike,
    x2: jax_ArrayLike,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not is_float_dtype(x1):
        x1 = x1.astype(default_float_dtype(as_native=True))
        x2 = x2.astype(default_float_dtype(as_native=True))
    return jnp.copysign(x1, x2)


def count_nonzero(
    a: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, list):
        axis = tuple(axis)
    if dtype is None:
        return jnp.count_nonzero(a, axis=axis, keepdims=keepdims)
    return jnp.array(jnp.count_nonzero(a, axis=axis, keepdims=keepdims), dtype=dtype)


def nansum(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int, ...], int]] = None,
    dtype: Optional[jnp.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return jnp.nansum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


def gcd(
    x1: Union[JaxArray, float, list, tuple],
    x2: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.gcd(x1, x2)


def isclose(
    a: JaxArray,
    b: JaxArray,
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def nan_to_num(
    x: JaxArray,
    /,
    *,
    copy: Optional[bool] = True,
    nan: Optional[Union[float, int]] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


def logaddexp2(
    x1: Union[JaxArray, float, list, tuple],
    x2: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not is_float_dtype(x1):
        x1 = x1.astype(default_float_dtype(as_native=True))
        x2 = x2.astype(default_float_dtype(as_native=True))
    return jnp.logaddexp2(x1, x2)


def signbit(
    x: Union[JaxArray, float, int, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.signbit(x)


def hypot(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.hypot(x1, x2)


def allclose(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> bool:
    return jnp.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def diff(
    x: JaxArray,
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[Union[JaxArray, int, float, list, tuple]] = None,
    append: Optional[Union[JaxArray, int, float, list, tuple]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x = jnp.asarray(x)
    if isinstance(prepend, (list, tuple)):
        prepend = jnp.asarray(prepend)
    if isinstance(append, (list, tuple)):
        append = jnp.asarray(append)
    return jnp.diff(x, n=n, axis=axis, prepend=prepend, append=append)


def fix(
    x: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fix(x, out=out)


def nextafter(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.nextafter(x1, x2)


def angle(
    z: JaxArray,
    /,
    *,
    deg: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.angle(z, deg=deg)


def imag(
    val: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.imag(val)


def zeta(
    x: JaxArray,
    q: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    temp = jnp.logical_and(jnp.greater(x, 0), jnp.equal(jnp.remainder(x, 2), 0))
    temp = jnp.logical_and(temp, jnp.less_equal(q, 0))
    temp = jnp.logical_and(temp, jnp.equal(jnp.remainder(q, 1), 0))
    inf_indices = jnp.logical_or(temp, jnp.equal(x, 1))
    temp = jnp.logical_and(jnp.not_equal(jnp.remainder(x, 2), 0), jnp.greater(x, 1))
    temp = jnp.logical_and(temp, jnp.less_equal(q, 0))
    nan_indices = jnp.logical_or(temp, jnp.less(x, 1))
    n, res = 1, 1 / q**x
    while n < 10000:
        term = 1 / (q + n) ** x
        n, res = n + 1, res + term
    ret = jnp.round(res, decimals=4)
    ret = ret.at[nan_indices].set(jnp.nan)
    ret = ret.at[inf_indices].set(jnp.inf)
    return ret


# def gradient(
#     x: JaxArray,
#     /,
#     *,
#     spacing: Optional[Union[int, list, tuple]] = 1,
#     axis: Optional[Union[int, list, tuple]] = None,
#     edge_order: Optional[int] = 1,
# ) -> Union[JaxArray, List[JaxArray]]:
#     if type(spacing) == int:
#         return jnp.gradient(x, spacing, axis=axis)
#     return jnp.gradient(x, *spacing, axis=axis)


def _normalize_axis_index(ax: int, ndim: int) -> int:
    if ax >= ndim or ax < -ndim:
        raise ValueError("axis index is out of range")
    return (ax + ndim) % ndim


def _normalize_axis_tuple(axis: Union[int, list, tuple], ndim: int) -> Tuple[int, ...]:
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    axis = tuple([_normalize_axis_index(ax, ndim) for ax in axis])
    if len(set(axis)) != len(axis):
        raise ValueError("repeated axis")
    return axis


def gradient(
    x: JaxArray,
    /,
    *,
    spacing: Optional[Union[int, list, tuple]] = 1,
    axis: Optional[Union[int, list, tuple]] = None,
    edge_order: Optional[int] = 1,
) -> Union[JaxArray, List[JaxArray]]:
    f = jnp.asarray(x)
    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = (
        -1
        if spacing is None
        else (0 if type(spacing) in (int, float) else len(spacing))
    )
    if n == -1:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [spacing] * len_axes
    elif n == 1 and jnp.ndim(spacing[0]) == 0:
        # single scalar for all axes
        dx = spacing * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(spacing)
        for i, distances in enumerate(dx):
            distances = jnp.asarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError(
                    "when 1d, distances must match "
                    "the length of the corresponding dimension"
                )
            if jnp.issubdtype(distances.dtype, jnp.integer):
                # Convert numpy integer types to float64 to avoid modular
                # arithmetic in np.diff(distances).
                distances = distances.astype(jnp.float64)
            diffx = jnp.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    otype = f.dtype
    if jnp.issubdtype(otype, jnp.integer):
        f = f.astype(jnp.float64)

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required."
            )
        # result allocation
        out = jnp.empty_like(f, dtype=otype)

        # spacing for the current axis
        uniform_spacing = jnp.ndim(ax_dx) == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        if uniform_spacing:
            out = out.at[tuple(slice1)].set(
                (f[tuple(slice4)] - f[tuple(slice2)]) / (2.0 * ax_dx)
            )
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2) / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = jnp.ones(N, dtype=int)
            # shape[axis] = -1
            shape = shape.at[axis].set(-1)
            jnp.reshape(a, shape)
            jnp.reshape(b, shape)
            jnp.reshape(c, shape)
            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out = out.at[tuple(slice1)].set(
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )
        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out = out.at[tuple(slice1)].set(
                (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0
            )

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out = out.at[tuple(slice1)].set(
                (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
            )

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2.0 / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out = out.at[tuple(slice1)].set(
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2.0 / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = -(dx2 + dx1) / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out = out.at[tuple(slice1)].set(
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals


def xlogy(x: JaxArray, y: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    x, y = promote_types_of_inputs(x, y)
    return js.special.xlogy(x, y)


def real(x: Union[JaxArray], /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.real(x)
