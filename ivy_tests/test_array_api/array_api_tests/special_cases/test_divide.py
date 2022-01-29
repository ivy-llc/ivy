"""
Special cases tests for divide.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import (NaN, assert_exactly_equal, assert_negative_mathematical_sign,
                             assert_positive_mathematical_sign, exactly_equal, greater, infinity,
                             isfinite, isnegative, ispositive, less, logical_and, logical_not,
                             logical_or, non_zero, same_sign, zero)
from ..hypothesis_helpers import numeric_arrays
from .._array_module import divide

from hypothesis import given


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_either(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If either `x1_i` or `x2_i` is `NaN`, the result is `NaN`.

    """
    res = divide(arg1, arg2)
    mask = logical_or(exactly_equal(arg1, NaN(arg1.shape, arg1.dtype)), exactly_equal(arg2, NaN(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_either__either_1(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is either `+infinity` or `-infinity` and `x2_i` is either `+infinity` or `-infinity`, the result is `NaN`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(logical_or(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype))), logical_or(exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype))))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_either__either_2(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is either `+0` or `-0` and `x2_i` is either `+0` or `-0`, the result is `NaN`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(logical_or(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg1, -zero(arg1.shape, arg1.dtype))), logical_or(exactly_equal(arg2, zero(arg2.shape, arg2.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype))))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__greater_1(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is greater than `0`, the result is `+0`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), greater(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__greater_2(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is greater than `0`, the result is `-0`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), greater(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__less_1(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is less than `0`, the result is `-0`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), less(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__less_2(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is less than `0`, the result is `+0`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), less(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_greater__equal_1(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is greater than `0` and `x2_i` is `+0`, the result is `+infinity`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_greater__equal_2(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is greater than `0` and `x2_i` is `-0`, the result is `-infinity`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(greater(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_less__equal_1(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is less than `0` and `x2_i` is `+0`, the result is `-infinity`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_less__equal_2(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is less than `0` and `x2_i` is `-0`, the result is `+infinity`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(less(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__equal_1(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is a positive (i.e., greater than `0`) finite number, the result is `+infinity`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), logical_and(isfinite(arg2), ispositive(arg2)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__equal_2(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is a negative (i.e., less than `0`) finite number, the result is `-infinity`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), logical_and(isfinite(arg2), isnegative(arg2)))
    assert_exactly_equal(res[mask], (-infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__equal_3(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is a positive (i.e., greater than `0`) finite number, the result is `-infinity`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), logical_and(isfinite(arg2), ispositive(arg2)))
    assert_exactly_equal(res[mask], (-infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__equal_4(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is a negative (i.e., less than `0`) finite number, the result is `+infinity`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), logical_and(isfinite(arg2), isnegative(arg2)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__equal_5(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is a positive (i.e., greater than `0`) finite number and `x2_i` is `+infinity`, the result is `+0`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(logical_and(isfinite(arg1), ispositive(arg1)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__equal_6(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is a positive (i.e., greater than `0`) finite number and `x2_i` is `-infinity`, the result is `-0`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(logical_and(isfinite(arg1), ispositive(arg1)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__equal_7(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is a negative (i.e., less than `0`) finite number and `x2_i` is `+infinity`, the result is `-0`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(logical_and(isfinite(arg1), isnegative(arg1)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_equal__equal_8(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` is a negative (i.e., less than `0`) finite number and `x2_i` is `-infinity`, the result is `+0`.

    """
    res = divide(arg1, arg2)
    mask = logical_and(logical_and(isfinite(arg1), isnegative(arg1)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_same_sign_both(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` and `x2_i` have the same mathematical sign and are both nonzero finite numbers, the result has a positive mathematical sign.

    """
    res = divide(arg1, arg2)
    mask = logical_and(same_sign(arg1, arg2), logical_and(logical_and(isfinite(arg1), non_zero(arg1)), logical_and(isfinite(arg2), non_zero(arg2))))
    assert_positive_mathematical_sign(res[mask])


@given(numeric_arrays, numeric_arrays)
def test_divide_special_cases_two_args_different_signs_both(arg1, arg2):
    """
    Special case test for `divide(x1, x2, /)`:

        -   If `x1_i` and `x2_i` have different mathematical signs and are both nonzero finite numbers, the result has a negative mathematical sign.

    """
    res = divide(arg1, arg2)
    mask = logical_and(logical_not(same_sign(arg1, arg2)), logical_and(logical_and(isfinite(arg1), non_zero(arg1)), logical_and(isfinite(arg2), non_zero(arg2))))
    assert_negative_mathematical_sign(res[mask])

# TODO: Implement REMAINING test for:
# -   In the remaining cases, where neither `-infinity`, `+0`, `-0`, nor `NaN` is involved, the quotient must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode. If the magnitude is too larger to represent, the operation overflows and the result is an `infinity` of appropriate mathematical sign. If the magnitude is too small to represent, the operation underflows and the result is a zero of appropriate mathematical sign.
