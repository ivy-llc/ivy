import pytest
from hypothesis import given, reject
from hypothesis import strategies as st

from .. import _array_module as xp
from .. import dtype_helpers as dh
from .. import shape_helpers as sh
from .. import xps
from ..test_creation_functions import frange
from ..test_manipulation_functions import roll_ndindex
from ..test_operators_and_elementwise_functions import (
    mock_int_dtype,
    oneway_broadcastable_shapes,
    oneway_promotable_dtypes,
)
from ..test_signatures import extension_module


def test_extension_module_is_extension():
    assert extension_module("linalg")


def test_extension_func_is_not_extension():
    assert not extension_module("linalg.cross")


@pytest.mark.parametrize(
    "r, size, elements",
    [
        (frange(0, 1, 1), 1, [0]),
        (frange(1, 0, -1), 1, [1]),
        (frange(0, 1, -1), 0, []),
        (frange(0, 1, 2), 1, [0]),
    ],
)
def test_frange(r, size, elements):
    assert len(r) == size
    assert list(r) == elements


@pytest.mark.parametrize(
    "shape, expected",
    [((), [()])],
)
def test_ndindex(shape, expected):
    assert list(sh.ndindex(shape)) == expected


@pytest.mark.parametrize(
    "shape, axis, expected",
    [
        ((1,), 0, [(slice(None, None),)]),
        ((1, 2), 0, [(slice(None, None), slice(None, None))]),
        (
            (2, 4),
            1,
            [(0, slice(None, None)), (1, slice(None, None))],
        ),
    ],
)
def test_axis_ndindex(shape, axis, expected):
    assert list(sh.axis_ndindex(shape, axis)) == expected


@pytest.mark.parametrize(
    "shape, axes, expected",
    [
        ((), (), [[()]]),
        ((1,), (0,), [[(0,)]]),
        (
            (2, 2),
            (0,),
            [
                [(0, 0), (1, 0)],
                [(0, 1), (1, 1)],
            ],
        ),
    ],
)
def test_axes_ndindex(shape, axes, expected):
    assert list(sh.axes_ndindex(shape, axes)) == expected


@pytest.mark.parametrize(
    "shape, shifts, axes, expected",
    [
        ((1, 1), (0,), (0,), [(0, 0)]),
        ((2, 1), (1, 1), (0, 1), [(1, 0), (0, 0)]),
        ((2, 2), (1, 1), (0, 1), [(1, 1), (1, 0), (0, 1), (0, 0)]),
        ((2, 2), (-1, 1), (0, 1), [(1, 1), (1, 0), (0, 1), (0, 0)]),
    ],
)
def test_roll_ndindex(shape, shifts, axes, expected):
    assert list(roll_ndindex(shape, shifts, axes)) == expected


@pytest.mark.parametrize(
    "idx, expected",
    [
        ((), "x"),
        (42, "x[42]"),
        ((42,), "x[42]"),
        (slice(None, 2), "x[:2]"),
        (slice(2, None), "x[2:]"),
        (slice(0, 2), "x[0:2]"),
        (slice(0, 2, -1), "x[0:2:-1]"),
        (slice(None, None, -1), "x[::-1]"),
        (slice(None, None), "x[:]"),
        (..., "x[...]"),
    ],
)
def test_fmt_idx(idx, expected):
    assert sh.fmt_idx("x", idx) == expected


@given(x=st.integers(), dtype=xps.unsigned_integer_dtypes() | xps.integer_dtypes())
def test_int_to_dtype(x, dtype):
    try:
        d = xp.asarray(x, dtype=dtype)
    except OverflowError:
        reject()
    assert mock_int_dtype(x, dtype) == d


@given(oneway_promotable_dtypes(dh.all_dtypes))
def test_oneway_promotable_dtypes(D):
    assert D.result_dtype == dh.result_type(*D)


@given(oneway_broadcastable_shapes())
def test_oneway_broadcastable_shapes(S):
    assert S.result_shape == sh.broadcast_shapes(*S)
