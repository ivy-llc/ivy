import math
from itertools import product
from typing import List, Sequence, Tuple, Union, get_args

import pytest
from hypothesis import assume, given, note
from hypothesis import strategies as st

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from . import xps
from .test_operators_and_elementwise_functions import oneway_promotable_dtypes
from .typing import DataType, Index, Param, Scalar, ScalarType, Shape

pytestmark = pytest.mark.ci


def scalar_objects(
    dtype: DataType, shape: Shape
) -> st.SearchStrategy[Union[Scalar, List[Scalar]]]:
    """Generates scalars or nested sequences which are valid for xp.asarray()"""
    size = math.prod(shape)
    return st.lists(xps.from_dtype(dtype), min_size=size, max_size=size).map(
        lambda l: sh.reshape(l, shape)
    )


def normalise_key(key: Index, shape: Shape) -> Tuple[Union[int, slice], ...]:
    """
    Normalise an indexing key.

    * If a non-tuple index, wrap as a tuple.
    * Represent ellipsis as equivalent slices.
    """
    _key = tuple(key) if isinstance(key, tuple) else (key,)
    if Ellipsis in _key:
        nonexpanding_key = tuple(i for i in _key if i is not None)
        start_a = nonexpanding_key.index(Ellipsis)
        stop_a = start_a + (len(shape) - (len(nonexpanding_key) - 1))
        slices = tuple(slice(None) for _ in range(start_a, stop_a))
        start_pos = _key.index(Ellipsis)
        _key = _key[:start_pos] + slices + _key[start_pos + 1 :]
    return _key


def get_indexed_axes_and_out_shape(
    key: Tuple[Union[int, slice, None], ...], shape: Shape
) -> Tuple[Tuple[Sequence[int], ...], Shape]:
    """
    From the (normalised) key and input shape, calculates:

    * indexed_axes: For each dimension, the axes which the key indexes.
    * out_shape: The resulting shape of indexing an array (of the input shape)
      with the key.
    """
    axes_indices = []
    out_shape = []
    a = 0
    for i in key:
        if i is None:
            out_shape.append(1)
        else:
            side = shape[a]
            if isinstance(i, int):
                if i < 0:
                    i += side
                axes_indices.append((i,))
            else:
                indices = range(side)[i]
                axes_indices.append(indices)
                out_shape.append(len(indices))
            a += 1
    return tuple(axes_indices), tuple(out_shape)


@given(shape=hh.shapes(), dtype=xps.scalar_dtypes(), data=st.data())
def test_getitem(shape, dtype, data):
    zero_sided = any(side == 0 for side in shape)
    if zero_sided:
        x = xp.zeros(shape, dtype=dtype)
    else:
        obj = data.draw(scalar_objects(dtype, shape), label="obj")
        x = xp.asarray(obj, dtype=dtype)
    note(f"{x=}")
    key = data.draw(xps.indices(shape=shape, allow_newaxis=True), label="key")

    out = x[key]

    ph.assert_dtype("__getitem__", x.dtype, out.dtype)
    _key = normalise_key(key, shape)
    axes_indices, out_shape = get_indexed_axes_and_out_shape(_key, shape)
    ph.assert_shape("__getitem__", out.shape, out_shape)
    out_zero_sided = any(side == 0 for side in out_shape)
    if not zero_sided and not out_zero_sided:
        out_obj = []
        for idx in product(*axes_indices):
            val = obj
            for i in idx:
                val = val[i]
            out_obj.append(val)
        out_obj = sh.reshape(out_obj, out_shape)
        expected = xp.asarray(out_obj, dtype=dtype)
        ph.assert_array_elements("__getitem__", out, expected)


@given(
    shape=hh.shapes(),
    dtypes=oneway_promotable_dtypes(dh.all_dtypes),
    data=st.data(),
)
def test_setitem(shape, dtypes, data):
    zero_sided = any(side == 0 for side in shape)
    if zero_sided:
        x = xp.zeros(shape, dtype=dtypes.result_dtype)
    else:
        obj = data.draw(scalar_objects(dtypes.result_dtype, shape), label="obj")
        x = xp.asarray(obj, dtype=dtypes.result_dtype)
    note(f"{x=}")
    key = data.draw(xps.indices(shape=shape), label="key")
    _key = normalise_key(key, shape)
    axes_indices, out_shape = get_indexed_axes_and_out_shape(_key, shape)
    value_strat = xps.arrays(dtype=dtypes.result_dtype, shape=out_shape)
    if out_shape == ():
        # We can pass scalars if we're only indexing one element
        value_strat |= xps.from_dtype(dtypes.result_dtype)
    value = data.draw(value_strat, label="value")

    res = xp.asarray(x, copy=True)
    res[key] = value

    ph.assert_dtype("__setitem__", x.dtype, res.dtype, repr_name="x.dtype")
    ph.assert_shape("__setitem__", res.shape, x.shape, repr_name="x.shape")
    f_res = sh.fmt_idx("x", key)
    if isinstance(value, get_args(Scalar)):
        msg = f"{f_res}={res[key]!r}, but should be {value=} [__setitem__()]"
        if math.isnan(value):
            assert xp.isnan(res[key]), msg
        else:
            assert res[key] == value, msg
    else:
        ph.assert_array_elements("__setitem__", res[key], value, out_repr=f_res)
    unaffected_indices = set(sh.ndindex(res.shape)) - set(product(*axes_indices))
    for idx in unaffected_indices:
        ph.assert_0d_equals(
            "__setitem__", f"old {f_res}", x[idx], f"modified {f_res}", res[idx]
        )


@pytest.mark.data_dependent_shapes
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
