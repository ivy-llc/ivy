"""
Special cases tests for add.

These tests are generated from the special cases listed in the spec.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

from ..array_helpers import (NaN, assert_exactly_equal, exactly_equal, infinity, isfinite,
                             logical_and, logical_or, non_zero, zero)
from ..hypothesis_helpers import numeric_arrays
from .._array_module import add

from hypothesis import given


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_either(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If either `x1_i` or `x2_i` is `NaN`, the result is `NaN`.

    """
    res = add(arg1, arg2)
    mask = logical_or(exactly_equal(arg1, NaN(arg1.shape, arg1.dtype)), exactly_equal(arg2, NaN(arg1.shape, arg1.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_1(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is `-infinity`, the result is `NaN`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_2(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is `+infinity`, the result is `NaN`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (NaN(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_3(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is `+infinity`, the result is `+infinity`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_4(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is `-infinity`, the result is `-infinity`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_5(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `+infinity` and `x2_i` is a finite number, the result is `+infinity`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, infinity(arg1.shape, arg1.dtype)), isfinite(arg2))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_6(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `-infinity` and `x2_i` is a finite number, the result is `-infinity`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -infinity(arg1.shape, arg1.dtype)), isfinite(arg2))
    assert_exactly_equal(res[mask], (-infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_7(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is a finite number and `x2_i` is `+infinity`, the result is `+infinity`.

    """
    res = add(arg1, arg2)
    mask = logical_and(isfinite(arg1), exactly_equal(arg2, infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_8(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is a finite number and `x2_i` is `-infinity`, the result is `-infinity`.

    """
    res = add(arg1, arg2)
    mask = logical_and(isfinite(arg1), exactly_equal(arg2, -infinity(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-infinity(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_9(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is `-0`, the result is `-0`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (-zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_10(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `-0` and `x2_i` is `+0`, the result is `+0`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, -zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_11(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is `-0`, the result is `+0`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_12(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is `+0` and `x2_i` is `+0`, the result is `+0`.

    """
    res = add(arg1, arg2)
    mask = logical_and(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg2, zero(arg2.shape, arg2.dtype)))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__equal_13(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is a nonzero finite number and `x2_i` is `-x1_i`, the result is `+0`.

    """
    res = add(arg1, arg2)
    mask = logical_and(logical_and(isfinite(arg1), non_zero(arg1)), exactly_equal(arg2, -arg1))
    assert_exactly_equal(res[mask], (zero(arg1.shape, arg1.dtype))[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_either__equal(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is either `+0` or `-0` and `x2_i` is a nonzero finite number, the result is `x2_i`.

    """
    res = add(arg1, arg2)
    mask = logical_and(logical_or(exactly_equal(arg1, zero(arg1.shape, arg1.dtype)), exactly_equal(arg1, -zero(arg1.shape, arg1.dtype))), logical_and(isfinite(arg2), non_zero(arg2)))
    assert_exactly_equal(res[mask], (arg2)[mask])


@given(numeric_arrays, numeric_arrays)
def test_add_special_cases_two_args_equal__either(arg1, arg2):
    """
    Special case test for `add(x1, x2, /)`:

        -   If `x1_i` is a nonzero finite number and `x2_i` is either `+0` or `-0`, the result is `x1_i`.

    """
    res = add(arg1, arg2)
    mask = logical_and(logical_and(isfinite(arg1), non_zero(arg1)), logical_or(exactly_equal(arg2, zero(arg2.shape, arg2.dtype)), exactly_equal(arg2, -zero(arg2.shape, arg2.dtype))))
    assert_exactly_equal(res[mask], (arg1)[mask])

# TODO: Implement REMAINING test for:
# -   In the remaining cases, when neither `infinity`, `+0`, `-0`, nor a `NaN` is involved, and the operands have the same mathematical sign or have different magnitudes, the sum must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported round mode. If the magnitude is too large to represent, the operation overflows and the result is an `infinity` of appropriate mathematical sign.
