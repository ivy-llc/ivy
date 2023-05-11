# global
import operator
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
from ivy.functional.backends.paddle.elementwise import _elementwise_helper
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


@with_supported_dtypes(
    {"2.4.2 and below": ("float64", "float32")},
    backend_version,
)
def trapz(
    y: paddle.Tensor,
    /,
    *,
    x: Optional[paddle.Tensor] = None,
    dx: Optional[float] = 1.0,
    axis: Optional[int] = -1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x is None:
        d = dx
    else:
        if x.ndim == 1:
            d = paddle.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = paddle.diff(x, axis=axis)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim

    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    with ivy.ArrayMode(False):
        if y.shape[axis] < 2:
            return ivy.zeros_like(ivy.squeeze(y, axis=axis))
        ret = ivy.sum(
            ivy.divide(
                ivy.multiply(
                    d,
                    ivy.add(
                        ivy.get_item(y, tuple(slice1)), ivy.get_item(y, tuple(slice2))
                    ),
                ),
                2.0,
            ),
            axis=axis,
        )

    return ret


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
    ret_dtype = x.dtype
    if x.dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
        x = x.cast("float32")
    prepend, append = [paddle.to_tensor(a, dtype=x.dtype) for a in [prepend, append]]
    return paddle.diff(x, n=n, axis=axis, prepend=prepend, append=append).cast(
        ret_dtype
    )


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
    with ivy.ArrayMode(False):
        return ivy.trunc(x)


def nextafter(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    with ivy.ArrayMode(False):
        eps = ivy.finfo(x1.dtype).eps
        return ivy.where(
            ivy.equal(x1, x2),
            x2,
            ivy.where(ivy.greater(x2, x1), ivy.add(x1, eps), ivy.subtract(x1, eps)),
        )


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


def _np_ndim(x):
    return ivy.array(x).ndim


@with_supported_dtypes(
    {"2.4.2 and below": ("float64", "float32")},
    backend_version,
)
def gradient(
    x: paddle.Tensor,
    /,
    *,
    spacing: Optional[Union[int, list, tuple]] = 1,
    axis: Optional[Union[int, list, tuple]] = None,
    edge_order: Optional[int] = 1,
) -> Union[paddle.Tensor, List[paddle.Tensor]]:
    """Https://github.com/numpy/numpy/blob/v1.23.0/numpy/lib/
    function_base.py#L969-L1312."""
    # TODO: Remove % x.shape[axis] once scatter_nd supports negative indices
    with ivy.ArrayMode(False):
        N = x.ndim  # number of dimensions
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
        elif n == 0:
            dx = [spacing] * len_axes
        elif n == 1 and _np_ndim(spacing[0]) == 0:
            # single scalar for all axes
            dx = spacing * len_axes
        elif n == len_axes:
            # scalar or 1d array for each axis
            dx = list(spacing)
            for i, distances in enumerate(dx):
                distances = paddle.to_tensor(distances)
                if _np_ndim(distances) == 0:
                    continue
                elif _np_ndim(distances) != 1:
                    raise ValueError("distances must be either scalars or 1d")
                if len(distances) != x.shape[axes[i]]:
                    raise ValueError(
                        "when 1d, distances must match "
                        "the length of the corresponding dimension {} {}".format(
                            len(distances), x.shape[axes[i]]
                        )
                    )
                if ivy.is_int_dtype(distances.dtype):
                    # Convert numpy integer types to float64 to avoid modular
                    # arithmetic in np.diff(distances).
                    distances = distances.astype("float64")
                diffx = ivy.diff(distances)
                # if distances are constant reduce to the scalar case
                # since it brings a consistent speedup
                # cmp = diffx == diffx[0]
                if ivy.all(ivy.equal(diffx, diffx[0])):
                    diffx = diffx[0]
                # if tf.reduce_sum(tf.cast(cmp, tf.int32)) == cmp.numel():
                #     print(diffx, (diffx == diffx[0]))
                #     diffx = diffx[0]
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

        if ivy.is_int_dtype(x.dtype):
            x = x.astype("float64")
        for axis, ax_dx in zip(axes, dx):
            if x.shape[axis] < edge_order + 1:
                raise ValueError(
                    "Shape of array too small to calculate a numerical gradient, "
                    "at least (edge_order + 1) elements are required."
                )
            # result allocation
            out = ivy.empty_like(x)  # x.clone()

            # spacing for the current axis
            uniform_spacing = _np_ndim(ax_dx) == 0

            # Numerical differentiation: 2nd order interior
            slice1[axis] = slice(1, -1)
            slice2[axis] = slice(None, -2)
            slice3[axis] = slice(1, -1)
            slice4[axis] = slice(2, None)
            if uniform_spacing:
                x_slice2 = ivy.get_item(x, tuple(slice2))
                x_slice4 = ivy.get_item(x, tuple(slice4))
                # since paddle doesn't support elementwise operations for empty tensors
                # numpy behaviour needs to be replicated manually
                if 0 not in x_slice2.shape + x_slice4.shape:
                    updates = ivy.divide(
                        ivy.subtract(x_slice2, x_slice4),
                        ivy.multiply(2.0, ax_dx),
                    )
                    ivy.scatter_nd(tuple(slice1), updates, reduction="replace", out=out)
            else:
                dx1 = ax_dx[0:-1]
                dx2 = ax_dx[1:]
                a = -(dx2) / (dx1 * (dx1 + dx2))
                b = (dx2 - dx1) / (dx1 * dx2)
                c = dx1 / (dx2 * (dx1 + dx2))
                ivy.scatter_nd(
                    tuple(slice1),
                    (
                        a * x[tuple(slice2)]
                        + b * x[tuple(slice3)]
                        + c * x[tuple(slice4)]
                    ),
                    reduction="replace",
                    out=out,
                )

            # Numerical differentiation: 1st order edges
            if edge_order == 1:
                slice1[axis] = 0
                slice2[axis] = 1
                slice3[axis] = 0
                dx_0 = ax_dx if uniform_spacing else ax_dx[0]
                # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
                x_slice2 = ivy.get_item(x, tuple(slice2))
                x_slice3 = ivy.get_item(x, tuple(slice3))
                updates = ivy.divide(ivy.subtract(x_slice2, x_slice3), dx_0)
                ivy.scatter_nd(
                    tuple(slice1),
                    updates,
                    reduction="replace",
                    out=out,
                )

                slice1[axis] = -1 % x.shape[axis]
                slice2[axis] = -1 % x.shape[axis]
                slice3[axis] = -2 % x.shape[axis]
                dx_n = ax_dx if uniform_spacing else ax_dx[-1]
                # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
                x_slice2 = ivy.get_item(x, tuple(slice2))
                x_slice3 = ivy.get_item(x, tuple(slice3))
                updates = ivy.divide(ivy.subtract(x_slice2, x_slice3), dx_n)
                ivy.scatter_nd(
                    tuple(slice1),
                    updates,
                    reduction="replace",
                    out=out,
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
                ivy.scatter_nd(
                    tuple(slice1),
                    (
                        a * x[tuple(slice2)]
                        + b * x[tuple(slice3)]
                        + c * x[tuple(slice4)]
                    ),
                    reduction="replace",
                    out=out,
                )

                slice1[axis] = -1 % x.shape[axis]
                slice2[axis] = -3 % x.shape[axis]
                slice3[axis] = -2 % x.shape[axis]
                slice4[axis] = -1 % x.shape[axis]
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
                ivy.scatter_nd(
                    tuple(slice1),
                    (
                        a * x[tuple(slice2)]
                        + b * x[tuple(slice3)]
                        + c * x[tuple(slice4)]
                    ),
                    reduction="replace",
                    out=out,
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


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def xlogy(
    x: paddle.Tensor, y: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x, y, ret_dtype = _elementwise_helper(x, y)
    with ivy.ArrayMode(False):
        x_ok = ivy.not_equal(x, 0.0)
        safe_x = ivy.where(x_ok, x, 1.0)
        safe_y = ivy.where(x_ok, y, 1.0)
        return ivy.where(
            x_ok, ivy.multiply(safe_x, ivy.log(safe_y)), ivy.zeros_like(x)
        ).cast(ret_dtype)


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
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def count_nonzero(
    a: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, list, tuple]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        return ivy.sum(ivy.not_equal(a, 0), axis=axis, keepdims=keepdims, dtype=dtype)


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
