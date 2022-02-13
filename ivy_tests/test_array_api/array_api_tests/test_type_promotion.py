"""
https://data-apis.github.io/array-api/latest/API_specification/type_promotion.html
"""
from collections import defaultdict
from typing import List, Tuple, Union

import pytest
from hypothesis import given, reject
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import xps
from .function_stubs import elementwise_functions
from .typing import DataType, Param, ScalarType

bitwise_shift_funcs = [
    "bitwise_left_shift",
    "bitwise_right_shift",
    "__lshift__",
    "__rshift__",
    "__ilshift__",
    "__irshift__",
]


# We pass kwargs to the elements strategy used by xps.arrays() so that we don't
# generate array elements that are erroneous or undefined for a function.
func_elements = defaultdict(
    lambda: None, {func: {"min_value": 1} for func in bitwise_shift_funcs}
)


def make_id(
    func_name: str,
    in_dtypes: Tuple[Union[DataType, ScalarType], ...],
    out_dtype: DataType,
) -> str:
    f_args = dh.fmt_types(in_dtypes)
    f_out_dtype = dh.dtype_to_name[out_dtype]
    return f"{func_name}({f_args}) -> {f_out_dtype}"


def mark_stubbed_dtypes(*dtypes):
    for dtype in dtypes:
        if isinstance(dtype, xp._UndefinedStub):
            return pytest.mark.skip(reason=f"xp.{dtype.name} not defined")
    else:
        return ()


func_params: List[Param[str, Tuple[DataType, ...], DataType]] = []
for func_name in elementwise_functions.__all__:
    valid_in_dtypes = dh.func_in_dtypes[func_name]
    ndtypes = ph.nargs(func_name)
    if ndtypes == 1:
        for in_dtype in valid_in_dtypes:
            out_dtype = xp.bool if dh.func_returns_bool[func_name] else in_dtype
            p = pytest.param(
                func_name,
                (in_dtype,),
                out_dtype,
                id=make_id(func_name, (in_dtype,), out_dtype),
                marks=mark_stubbed_dtypes(in_dtype, out_dtype),
            )
            func_params.append(p)
    elif ndtypes == 2:
        for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
            if in_dtype1 in valid_in_dtypes and in_dtype2 in valid_in_dtypes:
                out_dtype = (
                    xp.bool if dh.func_returns_bool[func_name] else promoted_dtype
                )
                p = pytest.param(
                    func_name,
                    (in_dtype1, in_dtype2),
                    out_dtype,
                    id=make_id(func_name, (in_dtype1, in_dtype2), out_dtype),
                    marks=mark_stubbed_dtypes(in_dtype1, in_dtype2, out_dtype),
                )
                func_params.append(p)
    else:
        raise NotImplementedError()


@pytest.mark.parametrize("func_name, in_dtypes, out_dtype", func_params)
@given(data=st.data())
def test_func_promotion(func_name, in_dtypes, out_dtype, data):
    func = getattr(xp, func_name)
    elements = func_elements[func_name]
    if len(in_dtypes) == 1:
        x = data.draw(
            xps.arrays(dtype=in_dtypes[0], shape=hh.shapes(), elements=elements),
            label="x",
        )
        out = func(x)
    else:
        arrays = []
        shapes = data.draw(
            hh.mutually_broadcastable_shapes(len(in_dtypes)), label="shapes"
        )
        for i, (dtype, shape) in enumerate(zip(in_dtypes, shapes), 1):
            x = data.draw(
                xps.arrays(dtype=dtype, shape=shape, elements=elements), label=f"x{i}"
            )
            arrays.append(x)
        try:
            out = func(*arrays)
        except OverflowError:
            reject()
    ph.assert_dtype(func_name, in_dtypes, out.dtype, out_dtype)


promotion_params: List[Param[Tuple[DataType, DataType], DataType]] = []
for (dtype1, dtype2), promoted_dtype in dh.promotion_table.items():
    p = pytest.param(
        (dtype1, dtype2),
        promoted_dtype,
        id=make_id("", (dtype1, dtype2), promoted_dtype),
        marks=mark_stubbed_dtypes(dtype1, dtype2, promoted_dtype),
    )
    promotion_params.append(p)


numeric_promotion_params = promotion_params[1:]


op_params: List[Param[str, str, Tuple[DataType, ...], DataType]] = []
op_to_symbol = {**dh.unary_op_to_symbol, **dh.binary_op_to_symbol}
for op, symbol in op_to_symbol.items():
    if op == "__matmul__":
        continue
    valid_in_dtypes = dh.func_in_dtypes[op]
    ndtypes = ph.nargs(op)
    if ndtypes == 1:
        for in_dtype in valid_in_dtypes:
            out_dtype = xp.bool if dh.func_returns_bool[op] else in_dtype
            p = pytest.param(
                op,
                f"{symbol}x",
                (in_dtype,),
                out_dtype,
                id=make_id(op, (in_dtype,), out_dtype),
                marks=mark_stubbed_dtypes(in_dtype, out_dtype),
            )
            op_params.append(p)
    else:
        for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
            if in_dtype1 in valid_in_dtypes and in_dtype2 in valid_in_dtypes:
                out_dtype = xp.bool if dh.func_returns_bool[op] else promoted_dtype
                p = pytest.param(
                    op,
                    f"x1 {symbol} x2",
                    (in_dtype1, in_dtype2),
                    out_dtype,
                    id=make_id(op, (in_dtype1, in_dtype2), out_dtype),
                    marks=mark_stubbed_dtypes(in_dtype1, in_dtype2, out_dtype),
                )
                op_params.append(p)
# We generate params for abs seperately as it does not have an associated symbol
for in_dtype in dh.func_in_dtypes["__abs__"]:
    p = pytest.param(
        "__abs__",
        "abs(x)",
        (in_dtype,),
        in_dtype,
        id=make_id("__abs__", (in_dtype,), in_dtype),
        marks=mark_stubbed_dtypes(in_dtype),
    )
    op_params.append(p)


@pytest.mark.parametrize("op, expr, in_dtypes, out_dtype", op_params)
@given(data=st.data())
def test_op_promotion(op, expr, in_dtypes, out_dtype, data):
    elements = func_elements[func_name]
    if len(in_dtypes) == 1:
        x = data.draw(
            xps.arrays(dtype=in_dtypes[0], shape=hh.shapes(), elements=elements),
            label="x",
        )
        out = eval(expr, {"x": x})
    else:
        locals_ = {}
        shapes = data.draw(
            hh.mutually_broadcastable_shapes(len(in_dtypes)), label="shapes"
        )
        for i, (dtype, shape) in enumerate(zip(in_dtypes, shapes), 1):
            locals_[f"x{i}"] = data.draw(
                xps.arrays(dtype=dtype, shape=shape, elements=elements), label=f"x{i}"
            )
        try:
            out = eval(expr, locals_)
        except OverflowError:
            reject()
    ph.assert_dtype(op, in_dtypes, out.dtype, out_dtype)


inplace_params: List[Param[str, str, Tuple[DataType, ...], DataType]] = []
for op, symbol in dh.inplace_op_to_symbol.items():
    if op == "__imatmul__":
        continue
    valid_in_dtypes = dh.func_in_dtypes[op]
    for (in_dtype1, in_dtype2), promoted_dtype in dh.promotion_table.items():
        if (
            in_dtype1 == promoted_dtype
            and in_dtype1 in valid_in_dtypes
            and in_dtype2 in valid_in_dtypes
        ):
            p = pytest.param(
                op,
                f"x1 {symbol} x2",
                (in_dtype1, in_dtype2),
                promoted_dtype,
                id=make_id(op, (in_dtype1, in_dtype2), promoted_dtype),
                marks=mark_stubbed_dtypes(in_dtype1, in_dtype2, promoted_dtype),
            )
            inplace_params.append(p)


@pytest.mark.parametrize("op, expr, in_dtypes, out_dtype", inplace_params)
@given(shape=hh.shapes(), data=st.data())
def test_inplace_op_promotion(op, expr, in_dtypes, out_dtype, shape, data):
    # TODO: test broadcastable shapes (that don't change x1's shape)
    elements = func_elements[func_name]
    x1 = data.draw(
        xps.arrays(dtype=in_dtypes[0], shape=shape, elements=elements), label="x1"
    )
    x2 = data.draw(
        xps.arrays(dtype=in_dtypes[1], shape=shape, elements=elements), label="x2"
    )
    locals_ = {"x1": x1, "x2": x2}
    try:
        exec(expr, locals_)
    except OverflowError:
        reject()
    x1 = locals_["x1"]
    ph.assert_dtype(op, in_dtypes, x1.dtype, out_dtype, repr_name="x1.dtype")


op_scalar_params: List[Param[str, str, DataType, ScalarType, DataType]] = []
for op, symbol in dh.binary_op_to_symbol.items():
    if op == "__matmul__":
        continue
    for in_dtype in dh.func_in_dtypes[op]:
        out_dtype = xp.bool if dh.func_returns_bool[op] else in_dtype
        for in_stype in dh.dtype_to_scalars[in_dtype]:
            p = pytest.param(
                op,
                f"x {symbol} s",
                in_dtype,
                in_stype,
                out_dtype,
                id=make_id(op, (in_dtype, in_stype), out_dtype),
                marks=mark_stubbed_dtypes(in_dtype, out_dtype),
            )
            op_scalar_params.append(p)


@pytest.mark.parametrize("op, expr, in_dtype, in_stype, out_dtype", op_scalar_params)
@given(data=st.data())
def test_op_scalar_promotion(op, expr, in_dtype, in_stype, out_dtype, data):
    elements = func_elements[func_name]
    kw = {k: in_stype is float for k in ("allow_nan", "allow_infinity")}
    s = data.draw(xps.from_dtype(in_dtype, **kw).map(in_stype), label="scalar")
    x = data.draw(
        xps.arrays(dtype=in_dtype, shape=hh.shapes(), elements=elements), label="x"
    )
    try:
        out = eval(expr, {"x": x, "s": s})
    except OverflowError:
        reject()
    ph.assert_dtype(op, [in_dtype, in_stype], out.dtype, out_dtype)


inplace_scalar_params: List[Param[str, str, DataType, ScalarType]] = []
for op, symbol in dh.inplace_op_to_symbol.items():
    if op == "__imatmul__":
        continue
    for dtype in dh.func_in_dtypes[op]:
        for in_stype in dh.dtype_to_scalars[dtype]:
            p = pytest.param(
                op,
                f"x {symbol} s",
                dtype,
                in_stype,
                id=make_id(op, (dtype, in_stype), dtype),
                marks=mark_stubbed_dtypes(dtype),
            )
            inplace_scalar_params.append(p)


@pytest.mark.parametrize("op, expr, dtype, in_stype", inplace_scalar_params)
@given(data=st.data())
def test_inplace_op_scalar_promotion(op, expr, dtype, in_stype, data):
    elements = func_elements[func_name]
    kw = {k: in_stype is float for k in ("allow_nan", "allow_infinity")}
    s = data.draw(xps.from_dtype(dtype, **kw).map(in_stype), label="scalar")
    x = data.draw(
        xps.arrays(dtype=dtype, shape=hh.shapes(), elements=elements), label="x"
    )
    locals_ = {"x": x, "s": s}
    try:
        exec(expr, locals_)
    except OverflowError:
        reject()
    x = locals_["x"]
    assert x.dtype == dtype, f"{x.dtype=!s}, but should be {dtype}"
    ph.assert_dtype(op, [dtype, in_stype], x.dtype, dtype, repr_name="x.dtype")


if __name__ == "__main__":
    for (i, j), p in dh.promotion_table.items():
        print(f"({i}, {j}) -> {p}")
