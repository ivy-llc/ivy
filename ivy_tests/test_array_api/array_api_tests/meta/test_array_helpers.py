from hypothesis import given, assume
from hypothesis.strategies import integers

from ..array_helpers import exactly_equal, notequal, int_to_dtype
from ..hypothesis_helpers import integer_dtypes
from ..dtype_helpers import dtype_nbits, dtype_signed
from .. import _array_module as xp

# TODO: These meta-tests currently only work with NumPy

def test_exactly_equal():
    a = xp.asarray([0, 0., -0., -0., xp.nan, xp.nan, 1, 1])
    b = xp.asarray([0, -1, -0.,  0., xp.nan,      1, 1, 2])

    res = xp.asarray([True, False, True, False, True, False, True, False])
    assert xp.all(xp.equal(exactly_equal(a, b), res))

def test_notequal():
    a = xp.asarray([0, 0., -0., -0., xp.nan, xp.nan, 1, 1])
    b = xp.asarray([0, -1, -0.,  0., xp.nan,      1, 1, 2])

    res = xp.asarray([False, True, False, False, False, True, False, True])
    assert xp.all(xp.equal(notequal(a, b), res))

@given(integers(), integer_dtypes)
def test_int_to_dtype(x, dtype):
    n = dtype_nbits[dtype]
    signed = dtype_signed[dtype]
    try:
        d = xp.asarray(x, dtype=dtype)
    except OverflowError:
        assume(False)
    assert int_to_dtype(x, n, signed) == d
