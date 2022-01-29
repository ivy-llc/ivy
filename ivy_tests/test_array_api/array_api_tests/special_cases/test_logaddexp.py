"""
Special cases tests for logaddexp.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import (NaN, assert_exactly_equal, exactly_equal, infinity, logical_and,
                             logical_not, logical_or)
from ..hypothesis_helpers import numeric_arrays
from .._array_module import logaddexp

from hypothesis import given


@given(numeric_arrays, numeric_arrays)
def test_logaddexp_special_cases_two_args_either(arg1, arg2):
    """
    Special case test for `logaddexp(x1, x2)`:

        -   If either `x1_i` or `x2_i` is `NaN`, the result is `NaN`.

    """
    res = logaddexp(arg1, arg2)
    mask = logical_or(exactly_equal(arg1, NaN(arg1.shape, arg1.dtype)), exactly_equal(arg2, NaN(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_logaddexp_special_cases_two_args_equal__notequal(arg1, arg2):
    """
    Special case test for `logaddexp(x1, x2)`:

        -   If `x1_i` is `+infinity` and `x2_i` is not `NaN`, the result is `+infinity`.

    """
    res = logaddexp(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), logical_not(exactly_equal(arg2, NaN(arg2.shape, arg2.dtype))))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_logaddexp_special_cases_two_args_notequal__equal(arg1, arg2):
    """
    Special case test for `logaddexp(x1, x2)`:

        -   If `x1_i` is not `NaN` and `x2_i` is `+infinity`, the result is `+infinity`.

    """
    res = logaddexp(arg1, arg2)
    mask = logical_and(logical_not(exactly_equal(arg1, NaN(arg1.shape, arg1.dtype))), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])
