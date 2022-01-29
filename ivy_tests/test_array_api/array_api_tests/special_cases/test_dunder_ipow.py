"""
Special cases tests for __ipow__.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from operator import ipow

from ..array_helpers import (NaN, asarray, assert_exactly_equal, exactly_equal, greater, infinity,
                             isfinite, isintegral, isodd, less, logical_and, logical_not, notequal,
                             one, zero)
from ..hypothesis_helpers import numeric_arrays

from hypothesis import given


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_notequal__equal(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is not equal to `1` and `x2_i` is `NaN`, the result is `NaN`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(logical_not(exactly_equal(arg1, one(arg1.shape, arg1.dtype))), exactly_equal(arg2, NaN(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_even_if_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x2_i` is `+0`, the result is `1`, even if `x1_i` is `NaN`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = exactly_equal(arg2, zero(arg2.shape, arg2.dtype))
    assert_exactly_equal(res[mask], (one(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_even_if_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x2_i` is `-0`, the result is `1`, even if `x1_i` is `NaN`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = exactly_equal(arg2, -zero(arg2.shape, arg2.dtype))
    assert_exactly_equal(res[mask], (one(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__notequal_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `NaN` and `x2_i` is not equal to `0`, the result is `NaN`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, NaN(arg1.shape, arg1.dtype)), notequal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__notequal_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `1` and `x2_i` is not `NaN`, the result is `1`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, one(arg1.shape, arg1.dtype)), logical_not(exactly_equal(arg2, NaN(arg2.shape, arg2.dtype))))
    assert_exactly_equal(res[mask], (one(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_absgreater__equal_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `abs(x1_i)` is greater than `1` and `x2_i` is `+infinity`, the result is `+infinity`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(greater(abs(arg1), one(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_absgreater__equal_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `abs(x1_i)` is greater than `1` and `x2_i` is `-infinity`, the result is `+0`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(greater(abs(arg1), one(arg1.shape, arg1.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_absequal__equal_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `abs(x1_i)` is `1` and `x2_i` is `+infinity`, the result is `1`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(abs(arg1), one(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (one(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_absequal__equal_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `abs(x1_i)` is `1` and `x2_i` is `-infinity`, the result is `1`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(abs(arg1), one(arg1.shape, arg1.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (one(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_absless__equal_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `abs(x1_i)` is less than `1` and `x2_i` is `+infinity`, the result is `+0`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(less(abs(arg1), one(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_absless__equal_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `abs(x1_i)` is less than `1` and `x2_i` is `-infinity`, the result is `+infinity`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(less(abs(arg1), one(arg1.shape, arg1.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__greater_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is greater than `0`, the result is `+infinity`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), greater(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__greater_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `+0` and `x2_i` is greater than `0`, the result is `+0`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), greater(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__less_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is less than `0`, the result is `+0`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), less(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__less_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `+0` and `x2_i` is less than `0`, the result is `+infinity`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), less(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__greater_equal_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `-infinity`, `x2_i` is greater than `0`, and `x2_i` is an odd integer value, the result is `-infinity`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), logical_and(greater(arg2, zero(arg2.shape, arg2.dtype)), isodd(arg2)))
    assert_exactly_equal(res[mask], (-infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__greater_equal_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `-0`, `x2_i` is greater than `0`, and `x2_i` is an odd integer value, the result is `-0`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), logical_and(greater(arg2, zero(arg2.shape, arg2.dtype)), isodd(arg2)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__greater_notequal_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `-infinity`, `x2_i` is greater than `0`, and `x2_i` is not an odd integer value, the result is `+infinity`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), logical_and(greater(arg2, zero(arg2.shape, arg2.dtype)), logical_not(isodd(arg2))))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__greater_notequal_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `-0`, `x2_i` is greater than `0`, and `x2_i` is not an odd integer value, the result is `+0`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), logical_and(greater(arg2, zero(arg2.shape, arg2.dtype)), logical_not(isodd(arg2))))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__less_equal_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `-infinity`, `x2_i` is less than `0`, and `x2_i` is an odd integer value, the result is `-0`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), logical_and(less(arg2, zero(arg2.shape, arg2.dtype)), isodd(arg2)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__less_equal_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `-0`, `x2_i` is less than `0`, and `x2_i` is an odd integer value, the result is `-infinity`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), logical_and(less(arg2, zero(arg2.shape, arg2.dtype)), isodd(arg2)))
    assert_exactly_equal(res[mask], (-infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__less_notequal_1(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `-infinity`, `x2_i` is less than `0`, and `x2_i` is not an odd integer value, the result is `+0`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), logical_and(less(arg2, zero(arg2.shape, arg2.dtype)), logical_not(isodd(arg2))))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_equal__less_notequal_2(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is `-0`, `x2_i` is less than `0`, and `x2_i` is not an odd integer value, the result is `+infinity`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), logical_and(less(arg2, zero(arg2.shape, arg2.dtype)), logical_not(isodd(arg2))))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_ipow_special_cases_two_args_less_equal__equal_notequal(arg1, arg2):
    """
    Special case test for `__ipow__(self, other, /)`:

        -   If `x1_i` is less than `0`, `x1_i` is a finite number, `x2_i` is a finite number, and `x2_i` is not an integer value, the result is `NaN`.

    """
    res = asarray(arg1, copy=True)
    ipow(res, arg2)
    mask = logical_and(logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), isfinite(arg1)), logical_and(isfinite(arg2), logical_not(isintegral(arg2))))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])
