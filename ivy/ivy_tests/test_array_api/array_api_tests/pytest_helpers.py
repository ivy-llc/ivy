import math
from inspect import getfullargspec
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from . import _array_module as xp
from . import dtype_helpers as dh
from . import shape_helpers as sh
from . import stubs
from .typing import Array, DataType, Scalar, ScalarType, Shape

__all__ = [
    "raises",
    "doesnt_raise",
    "nargs",
    "fmt_kw",
    "is_pos_zero",
    "is_neg_zero",
    "assert_dtype",
    "assert_kw_dtype",
    "assert_default_float",
    "assert_default_int",
    "assert_default_index",
    "assert_shape",
    "assert_result_shape",
    "assert_keepdimable_shape",
    "assert_0d_equals",
    "assert_fill",
    "assert_array_elements",
]


def raises(exceptions, function, message=""):
    """
    Like pytest.raises() except it allows custom error messages
    """
    try:
        function()
    except exceptions:
        return
    except Exception as e:
        if message:
            raise AssertionError(
                f"Unexpected exception {e!r} (expected {exceptions}): {message}"
            )
        raise AssertionError(f"Unexpected exception {e!r} (expected {exceptions})")
    raise AssertionError(message)


def doesnt_raise(function, message=""):
    """
    The inverse of raises().

    Use doesnt_raise(function) to test that function() doesn't raise any
    exceptions. Returns the result of calling function.
    """
    if not callable(function):
        raise ValueError("doesnt_raise should take a lambda")
    try:
        return function()
    except Exception as e:
        if message:
            raise AssertionError(f"Unexpected exception {e!r}: {message}")
        raise AssertionError(f"Unexpected exception {e!r}")


def nargs(func_name):
    return len(getfullargspec(stubs.name_to_func[func_name]).args)


def fmt_kw(kw: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in kw.items())


def is_pos_zero(n: float) -> bool:
    return n == 0 and math.copysign(1, n) == 1


def is_neg_zero(n: float) -> bool:
    return n == 0 and math.copysign(1, n) == -1


def assert_dtype(
    func_name: str,
    in_dtype: Union[DataType, Sequence[DataType]],
    out_dtype: DataType,
    expected: Optional[DataType] = None,
    *,
    repr_name: str = "out.dtype",
):
    """
    Assert the output dtype is as expected.

    If expected=None, we infer the expected dtype as in_dtype, to test
    out_dtype, e.g.

        >>> x = xp.arange(5, dtype=xp.uint8)
        >>> out = xp.abs(x)
        >>> assert_dtype('abs', x.dtype, out.dtype)

        is equivalent to

        >>> assert out.dtype == xp.uint8

    Or for multiple input dtypes, the expected dtype is inferred from their
    resulting type promotion, e.g.

        >>> x1 = xp.arange(5, dtype=xp.uint8)
        >>> x2 = xp.arange(5, dtype=xp.uint16)
        >>> out = xp.add(x1, x2)
        >>> assert_dtype('add', [x1.dtype, x2.dtype], out.dtype)

        is equivalent to

        >>> assert out.dtype == xp.uint16

    We can also specify the expected dtype ourselves, e.g.

        >>> x = xp.arange(5, dtype=xp.int8)
        >>> out = xp.sum(x)
        >>> default_int = xp.asarray(0).dtype
        >>> assert_dtype('sum', x, out.dtype, default_int)

    """
    in_dtypes = in_dtype if isinstance(in_dtype, Sequence) and not isinstance(in_dtype, str) else [in_dtype]
    f_in_dtypes = dh.fmt_types(tuple(in_dtypes))
    f_out_dtype = dh.dtype_to_name[out_dtype]
    if expected is None:
        expected = dh.result_type(*in_dtypes)
    f_expected = dh.dtype_to_name[expected]
    msg = (
        f"{repr_name}={f_out_dtype}, but should be {f_expected} "
        f"[{func_name}({f_in_dtypes})]"
    )
    assert out_dtype == expected, msg


def assert_kw_dtype(func_name: str, kw_dtype: DataType, out_dtype: DataType):
    """
    Assert the output dtype is the passed keyword dtype, e.g.

        >>> kw = {'dtype': xp.uint8}
        >>> out = xp.ones(5, **kw)
        >>> assert_kw_dtype('ones', kw['dtype'], out.dtype)

    """
    f_kw_dtype = dh.dtype_to_name[kw_dtype]
    f_out_dtype = dh.dtype_to_name[out_dtype]
    msg = (
        f"out.dtype={f_out_dtype}, but should be {f_kw_dtype} "
        f"[{func_name}(dtype={f_kw_dtype})]"
    )
    assert out_dtype == kw_dtype, msg


def assert_default_float(func_name: str, out_dtype: DataType):
    """
    Assert the output dtype is the default float, e.g.

        >>> out = xp.ones(5)
        >>> assert_default_float('ones', out.dtype)

    """
    f_dtype = dh.dtype_to_name[out_dtype]
    f_default = dh.dtype_to_name[dh.default_float]
    msg = (
        f"out.dtype={f_dtype}, should be default "
        f"floating-point dtype {f_default} [{func_name}()]"
    )
    assert out_dtype == dh.default_float, msg


def assert_default_int(func_name: str, out_dtype: DataType):
    """
    Assert the output dtype is the default int, e.g.

        >>> out = xp.full(5, 42)
        >>> assert_default_int('full', out.dtype)

    """
    f_dtype = dh.dtype_to_name[out_dtype]
    f_default = dh.dtype_to_name[dh.default_int]
    msg = (
        f"out.dtype={f_dtype}, should be default "
        f"integer dtype {f_default} [{func_name}()]"
    )
    assert out_dtype == dh.default_int, msg


def assert_default_index(func_name: str, out_dtype: DataType, repr_name="out.dtype"):
    """
    Assert the output dtype is the default index dtype, e.g.

        >>> out = xp.argmax(xp.arange(5))
        >>> assert_default_int('argmax', out.dtype)

    """
    f_dtype = dh.dtype_to_name[out_dtype]
    msg = (
        f"{repr_name}={f_dtype}, should be the default index dtype, "
        f"which is either int32 or int64 [{func_name}()]"
    )
    assert out_dtype in (xp.int32, xp.int64), msg


def assert_shape(
    func_name: str,
    out_shape: Union[int, Shape],
    expected: Union[int, Shape],
    /,
    repr_name="out.shape",
    **kw,
):
    """
    Assert the output shape is as expected, e.g.

        >>> out = xp.ones((3, 3, 3))
        >>> assert_shape('ones', out.shape, (3, 3, 3))

    """
    if isinstance(out_shape, int):
        out_shape = (out_shape,)
    if isinstance(expected, int):
        expected = (expected,)
    msg = (
        f"{repr_name}={out_shape}, but should be {expected} [{func_name}({fmt_kw(kw)})]"
    )
    assert out_shape == expected, msg


def assert_result_shape(
    func_name: str,
    in_shapes: Sequence[Shape],
    out_shape: Shape,
    /,
    expected: Optional[Shape] = None,
    *,
    repr_name="out.shape",
    **kw,
):
    """
    Assert the output shape is as expected.

    If expected=None, we infer the expected shape as the result of broadcasting
    in_shapes, to test against out_shape, e.g.

        >>> out = xp.add(xp.ones((3, 1)), xp.ones((1, 3)))
        >>> assert_shape('add', [(3, 1), (1, 3)], out.shape)

        is equivalent to

        >>> assert out.shape == (3, 3)

    """
    if expected is None:
        expected = sh.broadcast_shapes(*in_shapes)
    f_in_shapes = " . ".join(str(s) for s in in_shapes)
    f_sig = f" {f_in_shapes} "
    if kw:
        f_sig += f", {fmt_kw(kw)}"
    msg = f"{repr_name}={out_shape}, but should be {expected} [{func_name}({f_sig})]"
    assert out_shape == expected, msg


def assert_keepdimable_shape(
    func_name: str,
    in_shape: Shape,
    out_shape: Shape,
    axes: Tuple[int, ...],
    keepdims: bool,
    /,
    **kw,
):
    """
    Assert the output shape from a keepdimable function is as expected, e.g.

        >>> x = xp.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        >>> out1 = xp.max(x, keepdims=False)
        >>> out2 = xp.max(x, keepdims=True)
        >>> assert_keepdimable_shape('max', x.shape, out1.shape, (0, 1), False)
        >>> assert_keepdimable_shape('max', x.shape, out2.shape, (0, 1), True)

        is equivalent to

        >>> assert out1.shape == ()
        >>> assert out2.shape == (1, 1)

    """
    if keepdims:
        shape = tuple(1 if axis in axes else side for axis, side in enumerate(in_shape))
    else:
        shape = tuple(side for axis, side in enumerate(in_shape) if axis not in axes)
    assert_shape(func_name, out_shape, shape, **kw)


def assert_0d_equals(
    func_name: str, x_repr: str, x_val: Array, out_repr: str, out_val: Array, **kw
):
    """
    Assert a 0d array is as expected, e.g.

        >>> x = xp.asarray([0, 1, 2])
        >>> res = xp.asarray(x, copy=True)
        >>> res[0] = 42
        >>> assert_0d_equals('asarray', 'x[0]', x[0], 'x[0]', res[0])

        is equivalent to

        >>> assert res[0] == x[0]

    """
    msg = (
        f"{out_repr}={out_val}, but should be {x_repr}={x_val} "
        f"[{func_name}({fmt_kw(kw)})]"
    )
    if dh.is_float_dtype(out_val.dtype) and xp.isnan(out_val):
        assert xp.isnan(x_val), msg
    else:
        assert x_val == out_val, msg


def assert_scalar_equals(
    func_name: str,
    type_: ScalarType,
    idx: Shape,
    out: Scalar,
    expected: Scalar,
    /,
    repr_name: str = "out",
    **kw,
):
    """
    Assert a 0d array, convered to a scalar, is as expected, e.g.

        >>> x = xp.ones(5, dtype=xp.uint8)
        >>> out = xp.sum(x)
        >>> assert_scalar_equals('sum', int, (), int(out), 5)

        is equivalent to

        >>> assert int(out) == 5

    """
    repr_name = repr_name if idx == () else f"{repr_name}[{idx}]"
    f_func = f"{func_name}({fmt_kw(kw)})"
    if type_ in [bool, int]:
        msg = f"{repr_name}={out}, but should be {expected} [{f_func}]"
        assert out == expected, msg
    elif math.isnan(expected):
        msg = f"{repr_name}={out}, but should be {expected} [{f_func}]"
        assert math.isnan(out), msg
    else:
        msg = f"{repr_name}={out}, but should be roughly {expected} [{f_func}]"
        assert math.isclose(out, expected, rel_tol=0.25, abs_tol=1), msg


def assert_fill(
    func_name: str, fill_value: Scalar, dtype: DataType, out: Array, /, **kw
):
    """
    Assert all elements of an array is as expected, e.g.

        >>> out = xp.full(5, 42, dtype=xp.uint8)
        >>> assert_fill('full', 42, xp.uint8, out, 5)

        is equivalent to

        >>> assert xp.all(out == 42)

    """
    msg = f"out not filled with {fill_value} [{func_name}({fmt_kw(kw)})]\n{out=}"
    if math.isnan(fill_value):
        assert xp.all(xp.isnan(out)), msg
    else:
        assert xp.all(xp.equal(out, xp.asarray(fill_value, dtype=dtype))), msg


def assert_array_elements(
    func_name: str, out: Array, expected: Array, /, *, out_repr: str = "out", **kw
):
    """
    Assert array elements are (strictly) as expected, e.g.

        >>> x = xp.arange(5)
        >>> out = xp.asarray(x)
        >>> assert_array_elements('asarray', out, x)

        is equivalent to

        >>> assert xp.all(out == x)

    """
    dh.result_type(out.dtype, expected.dtype)  # sanity check
    assert_shape(func_name, out.shape, expected.shape, **kw)  # sanity check
    f_func = f"[{func_name}({fmt_kw(kw)})]"
    if dh.is_float_dtype(out.dtype):
        for idx in sh.ndindex(out.shape):
            at_out = out[idx]
            at_expected = expected[idx]
            msg = (
                f"{sh.fmt_idx(out_repr, idx)}={at_out}, should be {at_expected} "
                f"{f_func}"
            )
            if xp.isnan(at_expected):
                assert xp.isnan(at_out), msg
            elif at_expected == 0.0 or at_expected == -0.0:
                scalar_at_expected = float(at_expected)
                scalar_at_out = float(at_out)
                if is_pos_zero(scalar_at_expected):
                    assert is_pos_zero(scalar_at_out), msg
                else:
                    assert is_neg_zero(scalar_at_expected)  # sanity check
                    assert is_neg_zero(scalar_at_out), msg
            else:
                assert at_out == at_expected, msg
    else:
        assert xp.all(
            out == expected
        ), f"{out_repr} not as expected {f_func}\n{out_repr}={out!r}\n{expected=}"
