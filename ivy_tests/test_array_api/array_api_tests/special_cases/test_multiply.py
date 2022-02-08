"""
Special cases tests for multiply.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import (NaN, assert_exactly_equal, assert_isinf,
                             assert_negative_mathematical_sign, assert_positive_mathematical_sign,
                             exactly_equal, infinity, isfinite, logical_and, logical_not,
                             logical_or, non_zero, same_sign, zero)
from ..hypothesis_helpers import numeric_arrays
from .._array_module import multiply

from hypothesis import given


@given(numeric_arrays, numeric_arrays)
def test_multiply_special_cases_two_args_either(arg1, arg2):
    """
    Special case test for `multiply(x1, x2, /)`:

        -   If either `x1_i` or `x2_i` is `NaN`, the result is `NaN`.

    """
    res = multiply(arg1, arg2)
    mask = logical_or(exactly_equal(arg1, NaN(arg1.shape, arg1.dtype)), exactly_equal(arg2, NaN(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_multiply_special_cases_two_args_either__either_1(arg1, arg2):
    """
    Special case test for `multiply(x1, x2, /)`:

        -   If `x1_i` is either `+infinity` or `-infinity` and `x2_i` is either `+0` or `-0`, the result is `NaN`.

    """
    res = multiply(arg1, arg2)
    mask = logical_and(logical_or(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype))), logical_or(exactly_equal(arg2, zero(arg2.shape, arg2.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype))))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_multiply_special_cases_two_args_either__either_2(arg1, arg2):
    """
    Special case test for `multiply(x1, x2, /)`:

        -   If `x1_i` is either `+0` or `-0` and `x2_i` is either `+infinity` or `-infinity`, the result is `NaN`.

    """
    res = multiply(arg1, arg2)
    mask = logical_and(logical_or(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg1, -zero(arg1.shape, arg1.dtype))), logical_or(exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype))))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_multiply_special_cases_two_args_either__either_3(arg1, arg2):
    """
    Special case test for `multiply(x1, x2, /)`:

        -   If `x1_i` is either `+infinity` or `-infinity` and `x2_i` is either `+infinity` or `-infinity`, the result is a signed infinity with the mathematical sign determined by the rule already stated above.

    """
    res = multiply(arg1, arg2)
    mask = logical_and(logical_or(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype))), logical_or(exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype))))
    assert_isinf(res[mask])


@given(numeric_arrays, numeric_arrays)
def test_multiply_special_cases_two_args_same_sign_except(arg1, arg2):
    """
    Special case test for `multiply(x1, x2, /)`:

        -   If `x1_i` and `x2_i` have the same mathematical sign, the result has a positive mathematical sign, unless the result is `NaN`. If the result is `NaN`, the "sign" of `NaN` is implementation-defined.

    """
    res = multiply(arg1, arg2)
    mask = logical_and(same_sign(arg1, arg2), logical_not(exactly_equal(res, NaN(res.shape, res.dtype))))
    assert_positive_mathematical_sign(res[mask])


@given(numeric_arrays, numeric_arrays)
def test_multiply_special_cases_two_args_different_signs_except(arg1, arg2):
    """
    Special case test for `multiply(x1, x2, /)`:

        -   If `x1_i` and `x2_i` have different mathematical signs, the result has a negative mathematical sign, unless the result is `NaN`. If the result is `NaN`, the "sign" of `NaN` is implementation-defined.

    """
    res = multiply(arg1, arg2)
    mask = logical_and(logical_not(same_sign(arg1, arg2)), logical_not(exactly_equal(res, NaN(res.shape, res.dtype))))
    assert_negative_mathematical_sign(res[mask])


@given(numeric_arrays, numeric_arrays)
def test_multiply_special_cases_two_args_either__equal(arg1, arg2):
    """
    Special case test for `multiply(x1, x2, /)`:

        -   If `x1_i` is either `+infinity` or `-infinity` and `x2_i` is a nonzero finite number, the result is a signed infinity with the mathematical sign determined by the rule already stated above.

    """
    res = multiply(arg1, arg2)
    mask = logical_and(logical_or(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype))), logical_and(isfinite(arg2), non_zero(arg2)))
    assert_isinf(res[mask])


@given(numeric_arrays, numeric_arrays)
def test_multiply_special_cases_two_args_equal__either(arg1, arg2):
    """
    Special case test for `multiply(x1, x2, /)`:

        -   If `x1_i` is a nonzero finite number and `x2_i` is either `+infinity` or `-infinity`, the result is a signed infinity with the mathematical sign determined by the rule already stated above.

    """
    res = multiply(arg1, arg2)
    mask = logical_and(logical_and(isfinite(arg1), non_zero(arg1)), logical_or(exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype))))
    assert_isinf(res[mask])

# TODO: Implement REMAINING test for:
# -   In the remaining cases, where neither `infinity` nor `NaN` is involved, the product must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode. If the magnitude is too large to represent, the result is an `infinity` of appropriate mathematical sign. If the magnitude is too small to represent, the result is a zero of appropriate mathematical sign.
