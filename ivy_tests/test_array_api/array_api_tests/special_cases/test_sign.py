"""
Special cases tests for sign.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import (assert_exactly_equal, exactly_equal, greater, less, logical_or, one,
                             zero)
from ..hypothesis_helpers import numeric_arrays
from .._array_module import sign

from hypothesis import given


@given(numeric_arrays)
def test_sign_special_cases_one_arg_less(arg1):
    """
    Special case test for `sign(x, /)`:

        -   If `x_i` is less than `0`, the result is `-1`.

    """
    res = sign(arg1)
    mask = less(arg1, zero(arg1.shape, arg1.dtype))
    assert_exactly_equal(res[mask], (-one(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays)
def test_sign_special_cases_one_arg_either(arg1):
    """
    Special case test for `sign(x, /)`:

        -   If `x_i` is either `-0` or `+0`, the result is `0`.

    """
    res = sign(arg1)
    mask = logical_or(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), exactly_equal(arg1, zero(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays)
def test_sign_special_cases_one_arg_greater(arg1):
    """
    Special case test for `sign(x, /)`:

        -   If `x_i` is greater than `0`, the result is `+1`.

    """
    res = sign(arg1)
    mask = greater(arg1, zero(arg1.shape, arg1.dtype))
    assert_exactly_equal(res[mask], (one(arg1.shape, arg1.dtype))[mask])
