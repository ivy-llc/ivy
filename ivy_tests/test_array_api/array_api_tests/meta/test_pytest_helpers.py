from pytest import raises

from .. import _array_module as xp
from .. import pytest_helpers as ph


def test_assert_dtype():
    ph.assert_dtype("promoted_func", [xp.uint8, xp.int8], xp.int16)
    with raises(AssertionError):
        ph.assert_dtype("bad_func", [xp.uint8, xp.int8], xp.float32)
    ph.assert_dtype("bool_func", [xp.uint8, xp.int8], xp.bool, xp.bool)
    ph.assert_dtype("single_promoted_func", [xp.uint8], xp.uint8)
    ph.assert_dtype("single_bool_func", [xp.uint8], xp.bool, xp.bool)


def test_assert_array():
    ph.assert_array("int zeros", xp.asarray(0), xp.asarray(0))
    ph.assert_array("pos zeros", xp.asarray(0.0), xp.asarray(0.0))
    with raises(AssertionError):
        ph.assert_array("mixed sign zeros", xp.asarray(0.0), xp.asarray(-0.0))
    with raises(AssertionError):
        ph.assert_array("mixed sign zeros", xp.asarray(-0.0), xp.asarray(0.0))
