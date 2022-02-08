from pytest import raises

from .. import pytest_helpers as ph
from .. import _array_module as xp


def test_assert_dtype():
    ph.assert_dtype("promoted_func", (xp.uint8, xp.int8), xp.int16)
    with raises(AssertionError):
        ph.assert_dtype("bad_func", (xp.uint8, xp.int8), xp.float32)
    ph.assert_dtype("bool_func", (xp.uint8, xp.int8), xp.bool, xp.bool)
    ph.assert_dtype("single_promoted_func", (xp.uint8,), xp.uint8)
    ph.assert_dtype("single_bool_func", (xp.uint8,), xp.bool, xp.bool)
