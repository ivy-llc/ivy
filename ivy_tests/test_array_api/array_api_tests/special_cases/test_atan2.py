"""
Special cases tests for atan2.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import (NaN, assert_exactly_equal, exactly_equal, greater, infinity, isfinite,
                             less, logical_and, logical_or, zero, π)
from ..hypothesis_helpers import numeric_arrays
from .._array_module import atan2

from hypothesis import given


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_either(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If either `x1_i` or `x2_i` is `NaN`, the result is `NaN`.

    """
    res = atan2(arg1, arg2)
    mask = logical_or(exactly_equal(arg1, NaN(arg1.shape, arg1.dtype)), exactly_equal(arg2, NaN(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_greater__equal_1(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is greater than `0` and `x2_i` is `+0`, the result is an implementation-dependent approximation to `+π/2`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype)/2)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_greater__equal_2(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is greater than `0` and `x2_i` is `-0`, the result is an implementation-dependent approximation to `+π/2`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype)/2)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__greater_1(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is greater than `0`, the result is `+0`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), greater(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__greater_2(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is greater than `0`, the result is `-0`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), greater(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_1(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is `+0`, the result is `+0`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_2(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is `-0`, the result is an implementation-dependent approximation to `+π`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_3(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is `+0`, the result is `-0`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_4(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is `-0`, the result is an implementation-dependent approximation to `-π`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_5(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is finite, the result is an implementation-dependent approximation to `+π/2`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), isfinite(arg2))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype)/2)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_6(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is finite, the result is an implementation-dependent approximation to `-π/2`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), isfinite(arg2))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype)/2)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_7(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is `+infinity`, the result is an implementation-dependent approximation to `+π/4`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype)/4)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_8(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is `-infinity`, the result is an implementation-dependent approximation to `+3π/4`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+3*π(arg1.shape, arg1.dtype)/4)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_9(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is `+infinity`, the result is an implementation-dependent approximation to `-π/4`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype)/4)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__equal_10(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is `-infinity`, the result is an implementation-dependent approximation to `-3π/4`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-3*π(arg1.shape, arg1.dtype)/4)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__less_1(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is less than `0`, the result is an implementation-dependent approximation to `+π`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), less(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_equal__less_2(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is less than `0`, the result is an implementation-dependent approximation to `-π`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), less(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_less__equal_1(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is less than `0` and `x2_i` is `+0`, the result is an implementation-dependent approximation to `-π/2`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype)/2)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_less__equal_2(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is less than `0` and `x2_i` is `-0`, the result is an implementation-dependent approximation to `-π/2`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype)/2)[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_greater_equal__equal_1(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is greater than `0`, `x1_i` is a finite number, and `x2_i` is `+infinity`, the result is `+0`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), isfinite(arg1)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_greater_equal__equal_2(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is greater than `0`, `x1_i` is a finite number, and `x2_i` is `-infinity`, the result is an implementation-dependent approximation to `+π`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), isfinite(arg1)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (+π(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_less_equal__equal_1(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is less than `0`, `x1_i` is a finite number, and `x2_i` is `+infinity`, the result is `-0`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), isfinite(arg1)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_atan2_special_cases_two_args_less_equal__equal_2(arg1, arg2):
    """
    Special case test for `atan2(x1, x2, /)`:

        -   If `x1_i` is less than `0`, `x1_i` is a finite number, and `x2_i` is `-infinity`, the result is an implementation-dependent approximation to `-π`.

    """
    res = atan2(arg1, arg2)
    mask = logical_and(logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), isfinite(arg1)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-π(arg1.shape, arg1.dtype))[mask])
