"""
Special cases tests for acosh.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import NaN, assert_exactly_equal, exactly_equal, infinity, less, one, zero
from ..hypothesis_helpers import numeric_arrays
from .._array_module import acosh

from hypothesis import given


@given(numeric_arrays)
def test_acosh_special_cases_one_arg_equal_1(arg1):
    """
    Special case test for `acosh(x, /)`:

        -   If `x_i` is `NaN`, the result is `NaN`.

    """
    res = acosh(arg1)
    mask = exactly_equal(arg1, NaN(arg1.shape, arg1.dtype))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays)
def test_acosh_special_cases_one_arg_equal_2(arg1):
    """
    Special case test for `acosh(x, /)`:

        -   If `x_i` is `1`, the result is `+0`.

    """
    res = acosh(arg1)
    mask = exactly_equal(arg1, one(arg1.shape, arg1.dtype))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays)
def test_acosh_special_cases_one_arg_equal_3(arg1):
    """
    Special case test for `acosh(x, /)`:

        -   If `x_i` is `+infinity`, the result is `+infinity`.

    """
    res = acosh(arg1)
    mask = exactly_equal(arg1, infinity(arg1.shape, arg1.dtype))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays)
def test_acosh_special_cases_one_arg_less(arg1):
    """
    Special case test for `acosh(x, /)`:

        -   If `x_i` is less than `1`, the result is `NaN`.

    """
    res = acosh(arg1)
    mask = less(arg1, one(arg1.shape, arg1.dtype))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])
