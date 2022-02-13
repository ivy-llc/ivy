from .. import _array_module as xp
from ..array_helpers import exactly_equal, notequal

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

