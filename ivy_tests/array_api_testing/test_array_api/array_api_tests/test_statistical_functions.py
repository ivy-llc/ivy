import math
from typing import Optional

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.control import reject

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from ._array_module import _UndefinedStub
from .typing import DataType

pytestmark = pytest.mark.ci


def kwarg_dtypes(dtype: DataType) -> st.SearchStrategy[Optional[DataType]]:
    dtypes = [d2 for d1, d2 in dh.promotion_table if d1 == dtype]
    return st.none() | st.sampled_from(dtypes)


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_max(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.max(x, **kw)

    ph.assert_dtype("max", x.dtype, out.dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "max", x.shape, out.shape, _axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        max_ = scalar_type(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = max(elements)
        ph.assert_scalar_equals("max", scalar_type, out_idx, max_, expected)


@given(
    x=xps.arrays(
        dtype=xps.floating_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_mean(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.mean(x, **kw)

    ph.assert_dtype("mean", x.dtype, out.dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "mean", x.shape, out.shape, _axes, kw.get("keepdims", False), **kw
    )
    # Values testing mean is too finicky


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_min(x, data):
    kw = data.draw(hh.kwargs(axis=hh.axes(x.ndim), keepdims=st.booleans()), label="kw")

    out = xp.min(x, **kw)

    ph.assert_dtype("min", x.dtype, out.dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "min", x.shape, out.shape, _axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        min_ = scalar_type(out[out_idx])
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = min(elements)
        ph.assert_scalar_equals("min", scalar_type, out_idx, min_, expected)


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_prod(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=hh.axes(x.ndim),
            dtype=kwarg_dtypes(x.dtype),
            keepdims=st.booleans(),
        ),
        label="kw",
    )

    try:
        out = xp.prod(x, **kw)
    except OverflowError:
        reject()

    dtype = kw.get("dtype", None)
    if dtype is None:
        if dh.is_int_dtype(x.dtype):
            if x.dtype in dh.uint_dtypes:
                default_dtype = dh.default_uint
            else:
                default_dtype = dh.default_int
            m, M = dh.dtype_ranges[x.dtype]
            d_m, d_M = dh.dtype_ranges[default_dtype]
            if m < d_m or M > d_M:
                _dtype = x.dtype
            else:
                _dtype = default_dtype
        else:
            if dh.dtype_nbits[x.dtype] > dh.dtype_nbits[dh.default_float]:
                _dtype = x.dtype
            else:
                _dtype = dh.default_float
    else:
        _dtype = dtype
    # We ignore asserting the out dtype if what we expect is undefined
    # See https://github.com/data-apis/array-api-tests/issues/106
    if not isinstance(_dtype, _UndefinedStub):
        ph.assert_dtype("prod", x.dtype, out.dtype, _dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "prod", x.shape, out.shape, _axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        prod = scalar_type(out[out_idx])
        assume(math.isfinite(prod))
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = math.prod(elements)
        if dh.is_int_dtype(out.dtype):
            m, M = dh.dtype_ranges[out.dtype]
            assume(m <= expected <= M)
        ph.assert_scalar_equals("prod", scalar_type, out_idx, prod, expected)


@given(
    x=xps.arrays(
        dtype=xps.floating_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ).filter(lambda x: x.size >= 2),
    data=st.data(),
)
def test_std(x, data):
    axis = data.draw(hh.axes(x.ndim), label="axis")
    _axes = sh.normalise_axis(axis, x.ndim)
    N = sum(side for axis, side in enumerate(x.shape) if axis not in _axes)
    correction = data.draw(
        st.floats(0.0, N, allow_infinity=False, allow_nan=False) | st.integers(0, N),
        label="correction",
    )
    keepdims = data.draw(st.booleans(), label="keepdims")
    kw = data.draw(
        hh.specified_kwargs(
            ("axis", axis, None),
            ("correction", correction, 0.0),
            ("keepdims", keepdims, False),
        ),
        label="kw",
    )

    out = xp.std(x, **kw)

    ph.assert_dtype("std", x.dtype, out.dtype)
    ph.assert_keepdimable_shape(
        "std", x.shape, out.shape, _axes, kw.get("keepdims", False), **kw
    )
    # We can't easily test the result(s) as standard deviation methods vary a lot


@given(
    x=xps.arrays(
        dtype=xps.numeric_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_sum(x, data):
    kw = data.draw(
        hh.kwargs(
            axis=hh.axes(x.ndim),
            dtype=kwarg_dtypes(x.dtype),
            keepdims=st.booleans(),
        ),
        label="kw",
    )

    try:
        out = xp.sum(x, **kw)
    except OverflowError:
        reject()

    dtype = kw.get("dtype", None)
    if dtype is None:
        if dh.is_int_dtype(x.dtype):
            if x.dtype in dh.uint_dtypes:
                default_dtype = dh.default_uint
            else:
                default_dtype = dh.default_int
            m, M = dh.dtype_ranges[x.dtype]
            d_m, d_M = dh.dtype_ranges[default_dtype]
            if m < d_m or M > d_M:
                _dtype = x.dtype
            else:
                _dtype = default_dtype
        else:
            if dh.dtype_nbits[x.dtype] > dh.dtype_nbits[dh.default_float]:
                _dtype = x.dtype
            else:
                _dtype = dh.default_float
    else:
        _dtype = dtype
    ph.assert_dtype("sum", x.dtype, out.dtype, _dtype)
    _axes = sh.normalise_axis(kw.get("axis", None), x.ndim)
    ph.assert_keepdimable_shape(
        "sum", x.shape, out.shape, _axes, kw.get("keepdims", False), **kw
    )
    scalar_type = dh.get_scalar_type(out.dtype)
    for indices, out_idx in zip(sh.axes_ndindex(x.shape, _axes), sh.ndindex(out.shape)):
        sum_ = scalar_type(out[out_idx])
        assume(math.isfinite(sum_))
        elements = []
        for idx in indices:
            s = scalar_type(x[idx])
            elements.append(s)
        expected = sum(elements)
        if dh.is_int_dtype(out.dtype):
            m, M = dh.dtype_ranges[out.dtype]
            assume(m <= expected <= M)
        ph.assert_scalar_equals("sum", scalar_type, out_idx, sum_, expected)


@given(
    x=xps.arrays(
        dtype=xps.floating_dtypes(),
        shape=hh.shapes(min_side=1),
        elements={"allow_nan": False},
    ).filter(lambda x: x.size >= 2),
    data=st.data(),
)
def test_var(x, data):
    axis = data.draw(hh.axes(x.ndim), label="axis")
    _axes = sh.normalise_axis(axis, x.ndim)
    N = sum(side for axis, side in enumerate(x.shape) if axis not in _axes)
    correction = data.draw(
        st.floats(0.0, N, allow_infinity=False, allow_nan=False) | st.integers(0, N),
        label="correction",
    )
    keepdims = data.draw(st.booleans(), label="keepdims")
    kw = data.draw(
        hh.specified_kwargs(
            ("axis", axis, None),
            ("correction", correction, 0.0),
            ("keepdims", keepdims, False),
        ),
        label="kw",
    )

    out = xp.var(x, **kw)

    ph.assert_dtype("var", x.dtype, out.dtype)
    ph.assert_keepdimable_shape(
        "var", x.shape, out.shape, _axes, kw.get("keepdims", False), **kw
    )
    # We can't easily test the result(s) as variance methods vary a lot
