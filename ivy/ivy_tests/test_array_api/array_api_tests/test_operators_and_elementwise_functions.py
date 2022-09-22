import math
import operator
from enum import Enum, auto
from typing import Callable, List, NamedTuple, Optional, Sequence, TypeVar, Union

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.control import reject

from . import _array_module as xp
from . import array_helpers as ah
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .typing import Array, DataType, Param, Scalar, ScalarType, Shape

pytestmark = pytest.mark.ci


def all_integer_dtypes() -> st.SearchStrategy[DataType]:
    """Returns a strategy for signed and unsigned integer dtype objects."""
    return xps.unsigned_integer_dtypes() | xps.integer_dtypes()


def boolean_and_all_integer_dtypes() -> st.SearchStrategy[DataType]:
    """Returns a strategy for boolean and all integer dtype objects."""
    return xps.boolean_dtypes() | all_integer_dtypes()


class OnewayPromotableDtypes(NamedTuple):
    input_dtype: DataType
    result_dtype: DataType


@st.composite
def oneway_promotable_dtypes(
    draw, dtypes: Sequence[DataType]
) -> st.SearchStrategy[OnewayPromotableDtypes]:
    """Return a strategy for input dtypes that promote to result dtypes."""
    d1, d2 = draw(hh.mutually_promotable_dtypes(dtypes=dtypes))
    result_dtype = dh.result_type(d1, d2)
    if d1 == result_dtype:
        return OnewayPromotableDtypes(d2, d1)
    elif d2 == result_dtype:
        return OnewayPromotableDtypes(d1, d2)
    else:
        reject()


class OnewayBroadcastableShapes(NamedTuple):
    input_shape: Shape
    result_shape: Shape


@st.composite
def oneway_broadcastable_shapes(draw) -> st.SearchStrategy[OnewayBroadcastableShapes]:
    """Return a strategy for input shapes that broadcast to result shapes."""
    result_shape = draw(hh.shapes(min_side=1))
    input_shape = draw(
        xps.broadcastable_shapes(
            result_shape,
            # Override defaults so bad shapes are less likely to be generated.
            max_side=None if result_shape == () else max(result_shape),
            max_dims=len(result_shape),
        ).filter(lambda s: sh.broadcast_shapes(result_shape, s) == result_shape)
    )
    return OnewayBroadcastableShapes(input_shape, result_shape)


def mock_int_dtype(n: int, dtype: DataType) -> int:
    """Returns equivalent of `n` that mocks `dtype` behaviour."""
    nbits = dh.dtype_nbits[dtype]
    mask = (1 << nbits) - 1
    n &= mask
    if dh.dtype_signed[dtype]:
        highest_bit = 1 << (nbits - 1)
        if n & highest_bit:
            n = -((~n & mask) + 1)
    return n


# This module tests elementwise functions/operators against a reference
# implementation. We iterate through the input array(s) and resulting array,
# casting the indexed arrays to Python scalars and calculating the expected
# output with `refimpl` function.
#
# This is finicky to refactor, but possible and ultimately worthwhile - hence
# why these *_assert_again_refimpl() utilities exist.
#
# Values which are special-cased are generated and passed, but are filtered by
# the `filter_` callable before they can be asserted against `refimpl`. We
# automatically generate tests for special cases in the special_cases/ dir. We
# still pass them here so as to ensure their presence doesn't affect the outputs
# respective to non-special-cased elements.
#
# By default, results are casted to scalars the same way that the inputs are.
# You can specify a cast via `res_stype, i.e. when a function accepts numerical
# inputs but returns boolean arrays.
#
# By default, floating-point functions/methods are loosely asserted against. Use
# `strict_check=True` when they should be strictly asserted against, i.e.
# when a function should return intergrals. Likewise, use `strict_check=False`
# when integer function/methods should be loosely asserted against, i.e. when
# floats are used internally for optimisation or legacy reasons.


def isclose(a: float, b: float, rel_tol: float = 0.25, abs_tol: float = 1) -> bool:
    """Wraps math.isclose with very generous defaults.

    This is useful for many floating-point operations where the spec does not
    make accuracy requirements.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"{a=} and {b=}, but input must be finite")
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def default_filter(s: Scalar) -> bool:
    """Returns False when s is a non-finite or a signed zero.

    Used by default as these values are typically special-cased.
    """
    if isinstance(s, int):  # note bools are ints
        return True
    else:
        return math.isfinite(s) and s != 0


T = TypeVar("T")


def unary_assert_against_refimpl(
    func_name: str,
    in_: Array,
    res: Array,
    refimpl: Callable[[T], T],
    expr_template: Optional[str] = None,
    res_stype: Optional[ScalarType] = None,
    filter_: Callable[[Scalar], bool] = default_filter,
    strict_check: Optional[bool] = None,
):
    if in_.shape != res.shape:
        raise ValueError(f"{res.shape=}, but should be {in_.shape=}")
    if expr_template is None:
        expr_template = func_name + "({})={}"
    in_stype = dh.get_scalar_type(in_.dtype)
    if res_stype is None:
        res_stype = in_stype
    m, M = dh.dtype_ranges.get(res.dtype, (None, None))
    for idx in sh.ndindex(in_.shape):
        scalar_i = in_stype(in_[idx])
        if not filter_(scalar_i):
            continue
        try:
            expected = refimpl(scalar_i)
        except Exception:
            continue
        if res.dtype != xp.bool:
            assert m is not None and M is not None  # for mypy
            if expected <= m or expected >= M:
                continue
        scalar_o = res_stype(res[idx])
        f_i = sh.fmt_idx("x", idx)
        f_o = sh.fmt_idx("out", idx)
        expr = expr_template.format(f_i, expected)
        if strict_check == False or dh.is_float_dtype(res.dtype):
            assert isclose(scalar_o, expected), (
                f"{f_o}={scalar_o}, but should be roughly {expr} [{func_name}()]\n"
                f"{f_i}={scalar_i}"
            )
        else:
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be {expr} [{func_name}()]\n"
                f"{f_i}={scalar_i}"
            )


def binary_assert_against_refimpl(
    func_name: str,
    left: Array,
    right: Array,
    res: Array,
    refimpl: Callable[[T, T], T],
    expr_template: Optional[str] = None,
    res_stype: Optional[ScalarType] = None,
    left_sym: str = "x1",
    right_sym: str = "x2",
    res_name: str = "out",
    filter_: Callable[[Scalar], bool] = default_filter,
    strict_check: Optional[bool] = None,
):
    if expr_template is None:
        expr_template = func_name + "({}, {})={}"
    in_stype = dh.get_scalar_type(left.dtype)
    if res_stype is None:
        res_stype = in_stype
    m, M = dh.dtype_ranges.get(res.dtype, (None, None))
    for l_idx, r_idx, o_idx in sh.iter_indices(left.shape, right.shape, res.shape):
        scalar_l = in_stype(left[l_idx])
        scalar_r = in_stype(right[r_idx])
        if not (filter_(scalar_l) and filter_(scalar_r)):
            continue
        try:
            expected = refimpl(scalar_l, scalar_r)
        except Exception:
            continue
        if res.dtype != xp.bool:
            assert m is not None and M is not None  # for mypy
            if expected <= m or expected >= M:
                continue
        scalar_o = res_stype(res[o_idx])
        f_l = sh.fmt_idx(left_sym, l_idx)
        f_r = sh.fmt_idx(right_sym, r_idx)
        f_o = sh.fmt_idx(res_name, o_idx)
        expr = expr_template.format(f_l, f_r, expected)
        if strict_check == False or dh.is_float_dtype(res.dtype):
            assert isclose(scalar_o, expected), (
                f"{f_o}={scalar_o}, but should be roughly {expr} [{func_name}()]\n"
                f"{f_l}={scalar_l}, {f_r}={scalar_r}"
            )
        else:
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be {expr} [{func_name}()]\n"
                f"{f_l}={scalar_l}, {f_r}={scalar_r}"
            )


def right_scalar_assert_against_refimpl(
    func_name: str,
    left: Array,
    right: Scalar,
    res: Array,
    refimpl: Callable[[T, T], T],
    expr_template: str = None,
    res_stype: Optional[ScalarType] = None,
    left_sym: str = "x1",
    res_name: str = "out",
    filter_: Callable[[Scalar], bool] = default_filter,
    strict_check: Optional[bool] = None,
):
    if filter_(right):
        return  # short-circuit here as there will be nothing to test
    in_stype = dh.get_scalar_type(left.dtype)
    if res_stype is None:
        res_stype = in_stype
    m, M = dh.dtype_ranges.get(left.dtype, (None, None))
    for idx in sh.ndindex(res.shape):
        scalar_l = in_stype(left[idx])
        if not filter_(scalar_l):
            continue
        try:
            expected = refimpl(scalar_l, right)
        except Exception:
            continue
        if left.dtype != xp.bool:
            assert m is not None and M is not None  # for mypy
            if expected <= m or expected >= M:
                continue
        scalar_o = res_stype(res[idx])
        f_l = sh.fmt_idx(left_sym, idx)
        f_o = sh.fmt_idx(res_name, idx)
        expr = expr_template.format(f_l, right, expected)
        if strict_check == False or dh.is_float_dtype(res.dtype):
            assert isclose(scalar_o, expected), (
                f"{f_o}={scalar_o}, but should be roughly {expr} [{func_name}()]\n"
                f"{f_l}={scalar_l}"
            )
        else:
            assert scalar_o == expected, (
                f"{f_o}={scalar_o}, but should be {expr} [{func_name}()]\n"
                f"{f_l}={scalar_l}"
            )


# When appropiate, this module tests operators alongside their respective
# elementwise methods. We do this by parametrizing a generalised test method
# with every relevant method and operator.
#
# Notable arguments in the parameter's context object:
# - The function object, which for operator test cases is a wrapper that allows
#   test logic to be generalised.
# - The argument strategies, which can be used to draw arguments for the test
#   case. They may require additional filtering for certain test cases.
# - right_is_scalar (binary parameters only), which denotes if the right
#   argument is a scalar in a test case. This can be used to appropiately adjust
#   draw filtering and test logic.


func_to_op = {v: k for k, v in dh.op_to_func.items()}
all_op_to_symbol = {**dh.binary_op_to_symbol, **dh.inplace_op_to_symbol}
finite_kw = {"allow_nan": False, "allow_infinity": False}


class UnaryParamContext(NamedTuple):
    func_name: str
    func: Callable[[Array], Array]
    strat: st.SearchStrategy[Array]

    @property
    def id(self) -> str:
        return f"{self.func_name}"

    def __repr__(self):
        return f"UnaryParamContext(<{self.id}>)"


def make_unary_params(
    elwise_func_name: str, dtypes_strat: st.SearchStrategy[DataType]
) -> List[Param[UnaryParamContext]]:
    strat = xps.arrays(dtype=dtypes_strat, shape=hh.shapes())
    func_ctx = UnaryParamContext(
        func_name=elwise_func_name, func=getattr(xp, elwise_func_name), strat=strat
    )
    op_name = func_to_op[elwise_func_name]
    op_ctx = UnaryParamContext(
        func_name=op_name, func=lambda x: getattr(x, op_name)(), strat=strat
    )
    return [pytest.param(func_ctx, id=func_ctx.id), pytest.param(op_ctx, id=op_ctx.id)]


class FuncType(Enum):
    FUNC = auto()
    OP = auto()
    IOP = auto()


shapes_kw = {"min_side": 1}


class BinaryParamContext(NamedTuple):
    func_name: str
    func: Callable[[Array, Union[Scalar, Array]], Array]
    left_sym: str
    left_strat: st.SearchStrategy[Array]
    right_sym: str
    right_strat: st.SearchStrategy[Union[Scalar, Array]]
    right_is_scalar: bool
    res_name: str

    @property
    def id(self) -> str:
        return f"{self.func_name}({self.left_sym}, {self.right_sym})"

    def __repr__(self):
        return f"BinaryParamContext(<{self.id}>)"


def make_binary_params(
    elwise_func_name: str, dtypes: Sequence[DataType]
) -> List[Param[BinaryParamContext]]:
    if hh.FILTER_UNDEFINED_DTYPES:
        dtypes = [d for d in dtypes if not isinstance(d, xp._UndefinedStub)]
    shared_oneway_dtypes = st.shared(oneway_promotable_dtypes(dtypes))
    left_dtypes = shared_oneway_dtypes.map(lambda D: D.result_dtype)
    right_dtypes = shared_oneway_dtypes.map(lambda D: D.input_dtype)

    def make_param(
        func_name: str, func_type: FuncType, right_is_scalar: bool
    ) -> Param[BinaryParamContext]:
        if right_is_scalar:
            left_sym = "x"
            right_sym = "s"
        else:
            left_sym = "x1"
            right_sym = "x2"

        if right_is_scalar:
            left_strat = xps.arrays(dtype=left_dtypes, shape=hh.shapes(**shapes_kw))
            right_strat = right_dtypes.flatmap(lambda d: xps.from_dtype(d, **finite_kw))
        else:
            if func_type is FuncType.IOP:
                shared_oneway_shapes = st.shared(oneway_broadcastable_shapes())
                left_strat = xps.arrays(
                    dtype=left_dtypes,
                    shape=shared_oneway_shapes.map(lambda S: S.result_shape),
                )
                right_strat = xps.arrays(
                    dtype=right_dtypes,
                    shape=shared_oneway_shapes.map(lambda S: S.input_shape),
                )
            else:
                mutual_shapes = st.shared(
                    hh.mutually_broadcastable_shapes(2, **shapes_kw)
                )
                left_strat = xps.arrays(
                    dtype=left_dtypes, shape=mutual_shapes.map(lambda pair: pair[0])
                )
                right_strat = xps.arrays(
                    dtype=right_dtypes, shape=mutual_shapes.map(lambda pair: pair[1])
                )

        if func_type is FuncType.FUNC:
            func = getattr(xp, func_name)
        else:
            op_sym = all_op_to_symbol[func_name]
            expr = f"{left_sym} {op_sym} {right_sym}"
            if func_type is FuncType.OP:

                def func(l: Array, r: Union[Scalar, Array]) -> Array:
                    locals_ = {}
                    locals_[left_sym] = l
                    locals_[right_sym] = r
                    return eval(expr, locals_)

            else:

                def func(l: Array, r: Union[Scalar, Array]) -> Array:
                    locals_ = {}
                    locals_[left_sym] = xp.asarray(l, copy=True)  # prevents mutating l
                    locals_[right_sym] = r
                    exec(expr, locals_)
                    return locals_[left_sym]

            func.__name__ = func_name  # for repr

        if func_type is FuncType.IOP:
            res_name = left_sym
        else:
            res_name = "out"

        ctx = BinaryParamContext(
            func_name,
            func,
            left_sym,
            left_strat,
            right_sym,
            right_strat,
            right_is_scalar,
            res_name,
        )
        return pytest.param(ctx, id=ctx.id)

    op_name = func_to_op[elwise_func_name]
    params = [
        make_param(elwise_func_name, FuncType.FUNC, False),
        make_param(op_name, FuncType.OP, False),
        make_param(op_name, FuncType.OP, True),
    ]
    iop_name = f"__i{op_name[2:]}"
    if iop_name in dh.inplace_op_to_symbol.keys():
        params.append(make_param(iop_name, FuncType.IOP, False))
        params.append(make_param(iop_name, FuncType.IOP, True))

    return params


def binary_param_assert_dtype(
    ctx: BinaryParamContext,
    left: Array,
    right: Union[Array, Scalar],
    res: Array,
    expected: Optional[DataType] = None,
):
    if ctx.right_is_scalar:
        in_dtypes = left.dtype
    else:
        in_dtypes = [left.dtype, right.dtype]  # type: ignore
    ph.assert_dtype(
        ctx.func_name, in_dtypes, res.dtype, expected, repr_name=f"{ctx.res_name}.dtype"
    )


def binary_param_assert_shape(
    ctx: BinaryParamContext,
    left: Array,
    right: Union[Array, Scalar],
    res: Array,
    expected: Optional[Shape] = None,
):
    if ctx.right_is_scalar:
        in_shapes = [left.shape]
    else:
        in_shapes = [left.shape, right.shape]  # type: ignore
    ph.assert_result_shape(
        ctx.func_name, in_shapes, res.shape, expected, repr_name=f"{ctx.res_name}.shape"
    )


def binary_param_assert_against_refimpl(
    ctx: BinaryParamContext,
    left: Array,
    right: Union[Array, Scalar],
    res: Array,
    op_sym: str,
    refimpl: Callable[[T, T], T],
    res_stype: Optional[ScalarType] = None,
    filter_: Callable[[Scalar], bool] = default_filter,
    strict_check: Optional[bool] = None,
):
    expr_template = "({} " + op_sym + " {})={}"
    if ctx.right_is_scalar:
        right_scalar_assert_against_refimpl(
            func_name=ctx.func_name,
            left_sym=ctx.left_sym,
            left=left,
            right=right,
            res_stype=res_stype,
            res_name=ctx.res_name,
            res=res,
            refimpl=refimpl,
            expr_template=expr_template,
            filter_=filter_,
            strict_check=strict_check,
        )
    else:
        binary_assert_against_refimpl(
            func_name=ctx.func_name,
            left_sym=ctx.left_sym,
            left=left,
            right_sym=ctx.right_sym,
            right=right,
            res_stype=res_stype,
            res_name=ctx.res_name,
            res=res,
            refimpl=refimpl,
            expr_template=expr_template,
            filter_=filter_,
            strict_check=strict_check,
        )


@pytest.mark.parametrize("ctx", make_unary_params("abs", xps.numeric_dtypes()))
@given(data=st.data())
def test_abs(ctx, data):
    x = data.draw(ctx.strat, label="x")
    # abs of the smallest negative integer is out-of-scope
    if x.dtype in dh.int_dtypes:
        assume(xp.all(x > dh.dtype_ranges[x.dtype].min))

    out = ctx.func(x)

    ph.assert_dtype(ctx.func_name, x.dtype, out.dtype)
    ph.assert_shape(ctx.func_name, out.shape, x.shape)
    unary_assert_against_refimpl(
        ctx.func_name,
        x,
        out,
        abs,  # type: ignore
        expr_template="abs({})={}",
        filter_=lambda s: (
            s == float("infinity") or (math.isfinite(s) and not ph.is_neg_zero(s))
        ),
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_acos(x):
    out = xp.acos(x)
    ph.assert_dtype("acos", x.dtype, out.dtype)
    ph.assert_shape("acos", out.shape, x.shape)
    unary_assert_against_refimpl(
        "acos", x, out, math.acos, filter_=lambda s: default_filter(s) and -1 <= s <= 1
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_acosh(x):
    out = xp.acosh(x)
    ph.assert_dtype("acosh", x.dtype, out.dtype)
    ph.assert_shape("acosh", out.shape, x.shape)
    unary_assert_against_refimpl(
        "acosh", x, out, math.acosh, filter_=lambda s: default_filter(s) and s >= 1
    )


@pytest.mark.parametrize("ctx,", make_binary_params("add", dh.numeric_dtypes))
@given(data=st.data())
def test_add(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    try:
        res = ctx.func(left, right)
    except OverflowError:
        reject()

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    binary_param_assert_against_refimpl(ctx, left, right, res, "+", operator.add)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asin(x):
    out = xp.asin(x)
    ph.assert_dtype("asin", x.dtype, out.dtype)
    ph.assert_shape("asin", out.shape, x.shape)
    unary_assert_against_refimpl(
        "asin", x, out, math.asin, filter_=lambda s: default_filter(s) and -1 <= s <= 1
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_asinh(x):
    out = xp.asinh(x)
    ph.assert_dtype("asinh", x.dtype, out.dtype)
    ph.assert_shape("asinh", out.shape, x.shape)
    unary_assert_against_refimpl("asinh", x, out, math.asinh)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_atan(x):
    out = xp.atan(x)
    ph.assert_dtype("atan", x.dtype, out.dtype)
    ph.assert_shape("atan", out.shape, x.shape)
    unary_assert_against_refimpl("atan", x, out, math.atan)


@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_atan2(x1, x2):
    out = xp.atan2(x1, x2)
    ph.assert_dtype("atan2", [x1.dtype, x2.dtype], out.dtype)
    ph.assert_result_shape("atan2", [x1.shape, x2.shape], out.shape)
    binary_assert_against_refimpl("atan2", x1, x2, out, math.atan2)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_atanh(x):
    out = xp.atanh(x)
    ph.assert_dtype("atanh", x.dtype, out.dtype)
    ph.assert_shape("atanh", out.shape, x.shape)
    unary_assert_against_refimpl(
        "atanh",
        x,
        out,
        math.atanh,
        filter_=lambda s: default_filter(s) and -1 <= s <= 1,
    )


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_and", dh.bool_and_all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_and(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    res = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    if left.dtype == xp.bool:
        refimpl = operator.and_
    else:
        refimpl = lambda l, r: mock_int_dtype(l & r, res.dtype)
    binary_param_assert_against_refimpl(ctx, left, right, res, "&", refimpl)


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_left_shift", dh.all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_left_shift(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        assume(right >= 0)
    else:
        assume(not xp.any(ah.isnegative(right)))

    res = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    nbits = res.dtype
    binary_param_assert_against_refimpl(
        ctx, left, right, res, "<<", lambda l, r: l << r if r < nbits else 0
    )


@pytest.mark.parametrize(
    "ctx", make_unary_params("bitwise_invert", boolean_and_all_integer_dtypes())
)
@given(data=st.data())
def test_bitwise_invert(ctx, data):
    x = data.draw(ctx.strat, label="x")

    out = ctx.func(x)

    ph.assert_dtype(ctx.func_name, x.dtype, out.dtype)
    ph.assert_shape(ctx.func_name, out.shape, x.shape)
    if x.dtype == xp.bool:
        refimpl = operator.not_
    else:
        refimpl = lambda s: mock_int_dtype(~s, x.dtype)
    unary_assert_against_refimpl(ctx.func_name, x, out, refimpl, expr_template="~{}={}")


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_or", dh.bool_and_all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_or(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    res = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    if left.dtype == xp.bool:
        refimpl = operator.or_
    else:
        refimpl = lambda l, r: mock_int_dtype(l | r, res.dtype)
    binary_param_assert_against_refimpl(ctx, left, right, res, "|", refimpl)


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_right_shift", dh.all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_right_shift(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        assume(right >= 0)
    else:
        assume(not xp.any(ah.isnegative(right)))

    res = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    binary_param_assert_against_refimpl(
        ctx, left, right, res, ">>", lambda l, r: mock_int_dtype(l >> r, res.dtype)
    )


@pytest.mark.parametrize(
    "ctx", make_binary_params("bitwise_xor", dh.bool_and_all_int_dtypes)
)
@given(data=st.data())
def test_bitwise_xor(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    res = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    if left.dtype == xp.bool:
        refimpl = operator.xor
    else:
        refimpl = lambda l, r: mock_int_dtype(l ^ r, res.dtype)
    binary_param_assert_against_refimpl(ctx, left, right, res, "^", refimpl)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_ceil(x):
    out = xp.ceil(x)
    ph.assert_dtype("ceil", x.dtype, out.dtype)
    ph.assert_shape("ceil", out.shape, x.shape)
    unary_assert_against_refimpl("ceil", x, out, math.ceil, strict_check=True)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_cos(x):
    out = xp.cos(x)
    ph.assert_dtype("cos", x.dtype, out.dtype)
    ph.assert_shape("cos", out.shape, x.shape)
    unary_assert_against_refimpl("cos", x, out, math.cos)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_cosh(x):
    out = xp.cosh(x)
    ph.assert_dtype("cosh", x.dtype, out.dtype)
    ph.assert_shape("cosh", out.shape, x.shape)
    unary_assert_against_refimpl("cosh", x, out, math.cosh)


@pytest.mark.parametrize("ctx", make_binary_params("divide", dh.float_dtypes))
@given(data=st.data())
def test_divide(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        assume

    res = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    binary_param_assert_against_refimpl(
        ctx,
        left,
        right,
        res,
        "/",
        operator.truediv,
        filter_=lambda s: math.isfinite(s) and s != 0,
    )


@pytest.mark.parametrize("ctx", make_binary_params("equal", dh.all_dtypes))
@given(data=st.data())
def test_equal(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, out, xp.bool)
    binary_param_assert_shape(ctx, left, right, out)
    if not ctx.right_is_scalar:
        # We manually promote the dtypes as incorrect internal type promotion
        # could lead to false positives. For example
        #
        #     >>> xp.equal(
        #     ...     xp.asarray(1.0, dtype=xp.float32),
        #     ...     xp.asarray(1.00000001, dtype=xp.float64),
        #     ... )
        #
        # would erroneously be True if float64 downcasted to float32.
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        left = xp.astype(left, promoted_dtype)
        right = xp.astype(right, promoted_dtype)
    binary_param_assert_against_refimpl(
        ctx, left, right, out, "==", operator.eq, res_stype=bool
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_exp(x):
    out = xp.exp(x)
    ph.assert_dtype("exp", x.dtype, out.dtype)
    ph.assert_shape("exp", out.shape, x.shape)
    unary_assert_against_refimpl("exp", x, out, math.exp)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_expm1(x):
    out = xp.expm1(x)
    ph.assert_dtype("expm1", x.dtype, out.dtype)
    ph.assert_shape("expm1", out.shape, x.shape)
    unary_assert_against_refimpl("expm1", x, out, math.expm1)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_floor(x):
    out = xp.floor(x)
    ph.assert_dtype("floor", x.dtype, out.dtype)
    ph.assert_shape("floor", out.shape, x.shape)
    unary_assert_against_refimpl("floor", x, out, math.floor, strict_check=True)


@pytest.mark.parametrize("ctx", make_binary_params("floor_divide", dh.numeric_dtypes))
@given(data=st.data())
def test_floor_divide(ctx, data):
    left = data.draw(
        ctx.left_strat.filter(lambda x: not xp.any(x == 0)), label=ctx.left_sym
    )
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        assume(right != 0)
    else:
        assume(not xp.any(right == 0))

    res = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    binary_param_assert_against_refimpl(ctx, left, right, res, "//", operator.floordiv)


@pytest.mark.parametrize("ctx", make_binary_params("greater", dh.numeric_dtypes))
@given(data=st.data())
def test_greater(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, out, xp.bool)
    binary_param_assert_shape(ctx, left, right, out)
    if not ctx.right_is_scalar:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        left = xp.astype(left, promoted_dtype)
        right = xp.astype(right, promoted_dtype)
    binary_param_assert_against_refimpl(
        ctx, left, right, out, ">", operator.gt, res_stype=bool
    )


@pytest.mark.parametrize("ctx", make_binary_params("greater_equal", dh.numeric_dtypes))
@given(data=st.data())
def test_greater_equal(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, out, xp.bool)
    binary_param_assert_shape(ctx, left, right, out)
    if not ctx.right_is_scalar:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        left = xp.astype(left, promoted_dtype)
        right = xp.astype(right, promoted_dtype)
    binary_param_assert_against_refimpl(
        ctx, left, right, out, ">=", operator.ge, res_stype=bool
    )


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isfinite(x):
    out = xp.isfinite(x)
    ph.assert_dtype("isfinite", x.dtype, out.dtype, xp.bool)
    ph.assert_shape("isfinite", out.shape, x.shape)
    unary_assert_against_refimpl("isfinite", x, out, math.isfinite, res_stype=bool)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isinf(x):
    out = xp.isinf(x)
    ph.assert_dtype("isfinite", x.dtype, out.dtype, xp.bool)
    ph.assert_shape("isinf", out.shape, x.shape)
    unary_assert_against_refimpl("isinf", x, out, math.isinf, res_stype=bool)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_isnan(x):
    out = xp.isnan(x)
    ph.assert_dtype("isnan", x.dtype, out.dtype, xp.bool)
    ph.assert_shape("isnan", out.shape, x.shape)
    unary_assert_against_refimpl("isnan", x, out, math.isnan, res_stype=bool)


@pytest.mark.parametrize("ctx", make_binary_params("less", dh.numeric_dtypes))
@given(data=st.data())
def test_less(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, out, xp.bool)
    binary_param_assert_shape(ctx, left, right, out)
    if not ctx.right_is_scalar:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        left = xp.astype(left, promoted_dtype)
        right = xp.astype(right, promoted_dtype)
    binary_param_assert_against_refimpl(
        ctx, left, right, out, "<", operator.lt, res_stype=bool
    )


@pytest.mark.parametrize("ctx", make_binary_params("less_equal", dh.numeric_dtypes))
@given(data=st.data())
def test_less_equal(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, out, xp.bool)
    binary_param_assert_shape(ctx, left, right, out)
    if not ctx.right_is_scalar:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        left = xp.astype(left, promoted_dtype)
        right = xp.astype(right, promoted_dtype)
    binary_param_assert_against_refimpl(
        ctx, left, right, out, "<=", operator.le, res_stype=bool
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log(x):
    out = xp.log(x)
    ph.assert_dtype("log", x.dtype, out.dtype)
    ph.assert_shape("log", out.shape, x.shape)
    unary_assert_against_refimpl(
        "log", x, out, math.log, filter_=lambda s: default_filter(s) and s >= 1
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log1p(x):
    out = xp.log1p(x)
    ph.assert_dtype("log1p", x.dtype, out.dtype)
    ph.assert_shape("log1p", out.shape, x.shape)
    unary_assert_against_refimpl(
        "log1p", x, out, math.log1p, filter_=lambda s: default_filter(s) and s >= 1
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log2(x):
    out = xp.log2(x)
    ph.assert_dtype("log2", x.dtype, out.dtype)
    ph.assert_shape("log2", out.shape, x.shape)
    unary_assert_against_refimpl(
        "log2", x, out, math.log2, filter_=lambda s: default_filter(s) and s > 1
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_log10(x):
    out = xp.log10(x)
    ph.assert_dtype("log10", x.dtype, out.dtype)
    ph.assert_shape("log10", out.shape, x.shape)
    unary_assert_against_refimpl(
        "log10", x, out, math.log10, filter_=lambda s: default_filter(s) and s > 0
    )


def logaddexp(l: float, r: float) -> float:
    return math.log(math.exp(l) + math.exp(r))


@given(*hh.two_mutual_arrays(dh.float_dtypes))
def test_logaddexp(x1, x2):
    out = xp.logaddexp(x1, x2)
    ph.assert_dtype("logaddexp", [x1.dtype, x2.dtype], out.dtype)
    ph.assert_result_shape("logaddexp", [x1.shape, x2.shape], out.shape)
    binary_assert_against_refimpl("logaddexp", x1, x2, out, logaddexp)


@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_and(x1, x2):
    out = xp.logical_and(x1, x2)
    ph.assert_dtype("logical_and", [x1.dtype, x2.dtype], out.dtype)
    ph.assert_result_shape("logical_and", [x1.shape, x2.shape], out.shape)
    binary_assert_against_refimpl(
        "logical_and", x1, x2, out, operator.and_, expr_template="({} and {})={}"
    )


@given(xps.arrays(dtype=xp.bool, shape=hh.shapes()))
def test_logical_not(x):
    out = xp.logical_not(x)
    ph.assert_dtype("logical_not", x.dtype, out.dtype)
    ph.assert_shape("logical_not", out.shape, x.shape)
    unary_assert_against_refimpl(
        "logical_not", x, out, operator.not_, expr_template="(not {})={}"
    )


@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_or(x1, x2):
    out = xp.logical_or(x1, x2)
    ph.assert_dtype("logical_or", [x1.dtype, x2.dtype], out.dtype)
    ph.assert_result_shape("logical_or", [x1.shape, x2.shape], out.shape)
    binary_assert_against_refimpl(
        "logical_or", x1, x2, out, operator.or_, expr_template="({} or {})={}"
    )


@given(*hh.two_mutual_arrays([xp.bool]))
def test_logical_xor(x1, x2):
    out = xp.logical_xor(x1, x2)
    ph.assert_dtype("logical_xor", [x1.dtype, x2.dtype], out.dtype)
    ph.assert_result_shape("logical_xor", [x1.shape, x2.shape], out.shape)
    binary_assert_against_refimpl(
        "logical_xor", x1, x2, out, operator.xor, expr_template="({} ^ {})={}"
    )


@pytest.mark.parametrize("ctx", make_binary_params("multiply", dh.numeric_dtypes))
@given(data=st.data())
def test_multiply(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    res = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    binary_param_assert_against_refimpl(ctx, left, right, res, "*", operator.mul)


# TODO: clarify if uints are acceptable, adjust accordingly
@pytest.mark.parametrize(
    "ctx", make_unary_params("negative", xps.integer_dtypes() | xps.floating_dtypes())
)
@given(data=st.data())
def test_negative(ctx, data):
    x = data.draw(ctx.strat, label="x")
    # negative of the smallest negative integer is out-of-scope
    if x.dtype in dh.int_dtypes:
        assume(xp.all(x > dh.dtype_ranges[x.dtype].min))

    out = ctx.func(x)

    ph.assert_dtype(ctx.func_name, x.dtype, out.dtype)
    ph.assert_shape(ctx.func_name, out.shape, x.shape)
    unary_assert_against_refimpl(
        ctx.func_name, x, out, operator.neg, expr_template="-({})={}"  # type: ignore
    )


@pytest.mark.parametrize("ctx", make_binary_params("not_equal", dh.all_dtypes))
@given(data=st.data())
def test_not_equal(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    out = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, out, xp.bool)
    binary_param_assert_shape(ctx, left, right, out)
    if not ctx.right_is_scalar:
        # See test_equal note
        promoted_dtype = dh.promotion_table[left.dtype, right.dtype]
        left = xp.astype(left, promoted_dtype)
        right = xp.astype(right, promoted_dtype)
    binary_param_assert_against_refimpl(
        ctx, left, right, out, "!=", operator.ne, res_stype=bool
    )


@pytest.mark.parametrize("ctx", make_unary_params("positive", xps.numeric_dtypes()))
@given(data=st.data())
def test_positive(ctx, data):
    x = data.draw(ctx.strat, label="x")

    out = ctx.func(x)

    ph.assert_dtype(ctx.func_name, x.dtype, out.dtype)
    ph.assert_shape(ctx.func_name, out.shape, x.shape)
    ph.assert_array_elements(ctx.func_name, out, x)


@pytest.mark.parametrize("ctx", make_binary_params("pow", dh.numeric_dtypes))
@given(data=st.data())
def test_pow(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        if isinstance(right, int):
            assume(right >= 0)
    else:
        if dh.is_int_dtype(right.dtype):
            assume(xp.all(right >= 0))

    try:
        res = ctx.func(left, right)
    except OverflowError:
        reject()

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    # Values testing pow is too finicky


@pytest.mark.parametrize("ctx", make_binary_params("remainder", dh.numeric_dtypes))
@given(data=st.data())
def test_remainder(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)
    if ctx.right_is_scalar:
        assume(right != 0)
    else:
        assume(not xp.any(right == 0))

    res = ctx.func(left, right)

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    binary_param_assert_against_refimpl(ctx, left, right, res, "%", operator.mod)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_round(x):
    out = xp.round(x)
    ph.assert_dtype("round", x.dtype, out.dtype)
    ph.assert_shape("round", out.shape, x.shape)
    unary_assert_against_refimpl("round", x, out, round, strict_check=True)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes(), elements=finite_kw))
def test_sign(x):
    out = xp.sign(x)
    ph.assert_dtype("sign", x.dtype, out.dtype)
    ph.assert_shape("sign", out.shape, x.shape)
    unary_assert_against_refimpl(
        "sign", x, out, lambda s: math.copysign(1, s), filter_=lambda s: s != 0
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sin(x):
    out = xp.sin(x)
    ph.assert_dtype("sin", x.dtype, out.dtype)
    ph.assert_shape("sin", out.shape, x.shape)
    unary_assert_against_refimpl("sin", x, out, math.sin)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sinh(x):
    out = xp.sinh(x)
    ph.assert_dtype("sinh", x.dtype, out.dtype)
    ph.assert_shape("sinh", out.shape, x.shape)
    unary_assert_against_refimpl("sinh", x, out, math.sinh)


@given(xps.arrays(dtype=xps.numeric_dtypes(), shape=hh.shapes()))
def test_square(x):
    out = xp.square(x)
    ph.assert_dtype("square", x.dtype, out.dtype)
    ph.assert_shape("square", out.shape, x.shape)
    unary_assert_against_refimpl(
        "square", x, out, lambda s: s ** 2, expr_template="{}²={}"
    )


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_sqrt(x):
    out = xp.sqrt(x)
    ph.assert_dtype("sqrt", x.dtype, out.dtype)
    ph.assert_shape("sqrt", out.shape, x.shape)
    unary_assert_against_refimpl(
        "sqrt", x, out, math.sqrt, filter_=lambda s: default_filter(s) and s >= 0
    )


@pytest.mark.parametrize("ctx", make_binary_params("subtract", dh.numeric_dtypes))
@given(data=st.data())
def test_subtract(ctx, data):
    left = data.draw(ctx.left_strat, label=ctx.left_sym)
    right = data.draw(ctx.right_strat, label=ctx.right_sym)

    try:
        res = ctx.func(left, right)
    except OverflowError:
        reject()

    binary_param_assert_dtype(ctx, left, right, res)
    binary_param_assert_shape(ctx, left, right, res)
    binary_param_assert_against_refimpl(ctx, left, right, res, "-", operator.sub)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_tan(x):
    out = xp.tan(x)
    ph.assert_dtype("tan", x.dtype, out.dtype)
    ph.assert_shape("tan", out.shape, x.shape)
    unary_assert_against_refimpl("tan", x, out, math.tan)


@given(xps.arrays(dtype=xps.floating_dtypes(), shape=hh.shapes()))
def test_tanh(x):
    out = xp.tanh(x)
    ph.assert_dtype("tanh", x.dtype, out.dtype)
    ph.assert_shape("tanh", out.shape, x.shape)
    unary_assert_against_refimpl("tanh", x, out, math.tanh)


@given(xps.arrays(dtype=hh.numeric_dtypes, shape=xps.array_shapes()))
def test_trunc(x):
    out = xp.trunc(x)
    ph.assert_dtype("trunc", x.dtype, out.dtype)
    ph.assert_shape("trunc", out.shape, x.shape)
    unary_assert_against_refimpl("trunc", x, out, math.trunc, strict_check=True)
