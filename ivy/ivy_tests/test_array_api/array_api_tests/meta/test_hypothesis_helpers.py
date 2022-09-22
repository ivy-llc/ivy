from math import prod

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .. import _array_module as xp
from .. import array_helpers as ah
from .. import dtype_helpers as dh
from .. import hypothesis_helpers as hh
from .. import shape_helpers as sh
from .. import xps
from .._array_module import _UndefinedStub

UNDEFINED_DTYPES = any(isinstance(d, _UndefinedStub) for d in dh.all_dtypes)
pytestmark = [pytest.mark.skipif(UNDEFINED_DTYPES, reason="undefined dtypes")]

@given(hh.mutually_promotable_dtypes(dtypes=dh.float_dtypes))
def test_mutually_promotable_dtypes(pair):
    assert pair in (
        (xp.float32, xp.float32),
        (xp.float32, xp.float64),
        (xp.float64, xp.float32),
        (xp.float64, xp.float64),
    )


@given(
    hh.mutually_promotable_dtypes(
        dtypes=[xp.uint8, _UndefinedStub("uint16"), xp.uint32]
    )
)
def test_partial_mutually_promotable_dtypes(pair):
    assert pair in (
        (xp.uint8, xp.uint8),
        (xp.uint8, xp.uint32),
        (xp.uint32, xp.uint8),
        (xp.uint32, xp.uint32),
    )


def valid_shape(shape) -> bool:
    return (
        all(isinstance(side, int) for side in shape)
        and all(side >= 0 for side in shape)
        and prod(shape) < hh.MAX_ARRAY_SIZE
    )


@given(hh.shapes())
def test_shapes(shape):
    assert valid_shape(shape)


@given(hh.two_mutually_broadcastable_shapes)
def test_two_mutually_broadcastable_shapes(pair):
    for shape in pair:
        assert valid_shape(shape)


@given(hh.two_broadcastable_shapes())
def test_two_broadcastable_shapes(pair):
    for shape in pair:
        assert valid_shape(shape)
    assert sh.broadcast_shapes(pair[0], pair[1]) == pair[0]


@given(*hh.two_mutual_arrays())
def test_two_mutual_arrays(x1, x2):
    assert (x1.dtype, x2.dtype) in dh.promotion_table.keys()


def test_two_mutual_arrays_raises_on_bad_dtypes():
    with pytest.raises(TypeError):
        hh.two_mutual_arrays(dtypes=xps.scalar_dtypes())


def test_kwargs():
    results = []

    @given(hh.kwargs(n=st.integers(0, 10), c=st.from_regex("[a-f]")))
    @settings(max_examples=100)
    def run(kw):
        results.append(kw)
    run()

    assert all(isinstance(kw, dict) for kw in results)
    for size in [0, 1, 2]:
        assert any(len(kw) == size for kw in results)

    n_results = [kw for kw in results if "n" in kw]
    assert len(n_results) > 0
    assert all(isinstance(kw["n"], int) for kw in n_results)

    c_results = [kw for kw in results if "c" in kw]
    assert len(c_results) > 0
    assert all(isinstance(kw["c"], str) for kw in c_results)


def test_specified_kwargs():
    results = []

    @given(n=st.integers(0, 10), d=st.none() | xps.scalar_dtypes(), data=st.data())
    @settings(max_examples=100)
    def run(n, d, data):
        kw = data.draw(
            hh.specified_kwargs(
                hh.KVD("n", n, 0),
                hh.KVD("d", d, None),
            ),
            label="kw",
        )
        results.append(kw)
    run()

    assert all(isinstance(kw, dict) for kw in results)

    assert any(len(kw) == 0 for kw in results)

    assert any("n" not in kw.keys() for kw in results)
    assert any("n" in kw.keys() and kw["n"] == 0 for kw in results)
    assert any("n" in kw.keys() and kw["n"] != 0 for kw in results)

    assert any("d" not in kw.keys() for kw in results)
    assert any("d" in kw.keys() and kw["d"] is None for kw in results)
    assert any("d" in kw.keys() and kw["d"] is xp.float64 for kw in results)



@given(m=hh.symmetric_matrices(hh.shared_floating_dtypes,
                                     finite=st.shared(st.booleans(), key='finite')),
       dtype=hh.shared_floating_dtypes,
       finite=st.shared(st.booleans(), key='finite'))
def test_symmetric_matrices(m, dtype, finite):
    assert m.dtype == dtype
    # TODO: This part of this test should be part of the .mT test
    ah.assert_exactly_equal(m, m.mT)

    if finite:
        ah.assert_finite(m)

@given(m=hh.positive_definite_matrices(hh.shared_floating_dtypes),
       dtype=hh.shared_floating_dtypes)
def test_positive_definite_matrices(m, dtype):
    assert m.dtype == dtype
    # TODO: Test that it actually is positive definite
