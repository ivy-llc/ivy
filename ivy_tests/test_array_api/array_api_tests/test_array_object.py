import math
from itertools import product
from typing import List, get_args

import pytest
from hypothesis import assume, given, note
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .typing import DataType, Param, Scalar, ScalarType, Shape

pytestmark = pytest.mark.ci


def scalar_objects(dtype: DataType, shape: Shape) -> st.SearchStrategy[List[Scalar]]:
    """Generates scalars or nested sequences which are valid for xp.asarray()"""
    size = math.prod(shape)
    return st.lists(xps.from_dtype(dtype), min_size=size, max_size=size).map(
        lambda l: sh.reshape(l, shape)
    )


@given(hh.shapes(min_side=1), st.data())  # TODO: test 0-sided arrays
def test_getitem(shape, data):
    dtype = data.draw(xps.scalar_dtypes(), label="dtype")
    obj = data.draw(scalar_objects(dtype, shape), label="obj")
    x = xp.asarray(obj, dtype=dtype)
    note(f"{x=}")
    key = data.draw(xps.indices(shape=shape), label="key")

    out = x[key]

    ph.assert_dtype("__getitem__", x.dtype, out.dtype)
    _key = tuple(key) if isinstance(key, tuple) else (key,)
    if Ellipsis in _key:
        start_a = _key.index(Ellipsis)
        stop_a = start_a + (len(shape) - (len(_key) - 1))
        slices = tuple(slice(None, None) for _ in range(start_a, stop_a))
        _key = _key[:start_a] + slices + _key[start_a + 1 :]
    axes_indices = []
    out_shape = []
    for a, i in enumerate(_key):
        if isinstance(i, int):
            axes_indices.append([i])
        else:
            side = shape[a]
            indices = range(side)[i]
            axes_indices.append(indices)
            out_shape.append(len(indices))
    out_shape = tuple(out_shape)
    ph.assert_shape("__getitem__", out.shape, out_shape)
    assume(all(len(indices) > 0 for indices in axes_indices))
    out_obj = []
    for idx in product(*axes_indices):
        val = obj
        for i in idx:
            val = val[i]
        out_obj.append(val)
    out_obj = sh.reshape(out_obj, out_shape)
    expected = xp.asarray(out_obj, dtype=dtype)
    ph.assert_array("__getitem__", out, expected)


@given(hh.shapes(min_side=1), st.data())  # TODO: test 0-sided arrays
def test_setitem(shape, data):
    dtype = data.draw(xps.scalar_dtypes(), label="dtype")
    obj = data.draw(scalar_objects(dtype, shape), label="obj")
    x = xp.asarray(obj, dtype=dtype)
    note(f"{x=}")
    # TODO: test setting non-0d arrays
    key = data.draw(xps.indices(shape=shape, max_dims=0), label="key")
    value = data.draw(
        xps.from_dtype(dtype) | xps.arrays(dtype=dtype, shape=()), label="value"
    )

    res = xp.asarray(x, copy=True)
    res[key] = value

    ph.assert_dtype("__setitem__", x.dtype, res.dtype, repr_name="x.dtype")
    ph.assert_shape("__setitem__", res.shape, x.shape, repr_name="x.shape")
    if isinstance(value, get_args(Scalar)):
        msg = f"x[{key}]={res[key]!r}, but should be {value=} [__setitem__()]"
        if math.isnan(value):
            assert xp.isnan(res[key]), msg
        else:
            assert res[key] == value, msg
    else:
        ph.assert_0d_equals(
            "__setitem__", "value", value, f"modified x[{key}]", res[key]
        )
    _key = key if isinstance(key, tuple) else (key,)
    assume(all(isinstance(i, int) for i in _key))  # TODO: normalise slices and ellipsis
    _key = tuple(i if i >= 0 else s + i for i, s in zip(_key, x.shape))
    unaffected_indices = list(sh.ndindex(res.shape))
    unaffected_indices.remove(_key)
    for idx in unaffected_indices:
        ph.assert_0d_equals(
            "__setitem__", f"old x[{idx}]", x[idx], f"modified x[{idx}]", res[idx]
        )


# TODO: make mask tests optional


@given(hh.shapes(), st.data())
def test_getitem_masking(shape, data):
    x = data.draw(xps.arrays(xps.scalar_dtypes(), shape=shape), label="x")
    mask_shapes = st.one_of(
        st.sampled_from([x.shape, ()]),
        st.lists(st.booleans(), min_size=x.ndim, max_size=x.ndim).map(
            lambda l: tuple(s if b else 0 for s, b in zip(x.shape, l))
        ),
        hh.shapes(),
    )
    key = data.draw(xps.arrays(dtype=xp.bool, shape=mask_shapes), label="key")

    if key.ndim > x.ndim or not all(
        ks in (xs, 0) for xs, ks in zip(x.shape, key.shape)
    ):
        with pytest.raises(IndexError):
            x[key]
        return

    out = x[key]

    ph.assert_dtype("__getitem__", x.dtype, out.dtype)
    if key.ndim == 0:
        out_shape = (1,) if key else (0,)
        out_shape += x.shape
    else:
        size = int(xp.sum(xp.astype(key, xp.uint8)))
        out_shape = (size,) + x.shape[key.ndim :]
    ph.assert_shape("__getitem__", out.shape, out_shape)
    if not any(s == 0 for s in key.shape):
        assume(key.ndim == x.ndim)  # TODO: test key.ndim < x.ndim scenarios
        out_indices = sh.ndindex(out.shape)
        for x_idx in sh.ndindex(x.shape):
            if key[x_idx]:
                out_idx = next(out_indices)
                ph.assert_0d_equals(
                    "__getitem__",
                    f"x[{x_idx}]",
                    x[x_idx],
                    f"out[{out_idx}]",
                    out[out_idx],
                )


@given(hh.shapes(), st.data())
def test_setitem_masking(shape, data):
    x = data.draw(xps.arrays(xps.scalar_dtypes(), shape=shape), label="x")
    key = data.draw(xps.arrays(dtype=xp.bool, shape=shape), label="key")
    value = data.draw(
        xps.from_dtype(x.dtype) | xps.arrays(dtype=x.dtype, shape=()), label="value"
    )

    res = xp.asarray(x, copy=True)
    res[key] = value

    ph.assert_dtype("__setitem__", x.dtype, res.dtype, repr_name="x.dtype")
    ph.assert_shape("__setitem__", res.shape, x.shape, repr_name="x.dtype")
    scalar_type = dh.get_scalar_type(x.dtype)
    for idx in sh.ndindex(x.shape):
        if key[idx]:
            if isinstance(value, get_args(Scalar)):
                ph.assert_scalar_equals(
                    "__setitem__",
                    scalar_type,
                    idx,
                    scalar_type(res[idx]),
                    value,
                    repr_name="modified x",
                )
            else:
                ph.assert_0d_equals(
                    "__setitem__", "value", value, f"modified x[{idx}]", res[idx]
                )
        else:
            ph.assert_0d_equals(
                "__setitem__", f"old x[{idx}]", x[idx], f"modified x[{idx}]", res[idx]
            )


def make_param(method_name: str, dtype: DataType, stype: ScalarType) -> Param:
    return pytest.param(
        method_name, dtype, stype, id=f"{method_name}({dh.dtype_to_name[dtype]})"
    )


@pytest.mark.parametrize(
    "method_name, dtype, stype",
    [make_param("__bool__", xp.bool, bool)]
    + [make_param("__int__", d, int) for d in dh.all_int_dtypes]
    + [make_param("__index__", d, int) for d in dh.all_int_dtypes]
    + [make_param("__float__", d, float) for d in dh.float_dtypes],
)
@given(data=st.data())
def test_scalar_casting(method_name, dtype, stype, data):
    x = data.draw(xps.arrays(dtype, shape=()), label="x")
    method = getattr(x, method_name)
    out = method()
    assert isinstance(
        out, stype
    ), f"{method_name}({x})={out}, which is not a {stype.__name__} scalar"
