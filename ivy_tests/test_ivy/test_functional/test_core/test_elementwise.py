"""Collection of tests for elementwise functions."""

from numbers import Number

# global
import numpy as np
from hypothesis import given, assume

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

_zero = np.asarray(0, dtype="uint8")
_one = np.asarray(1, dtype="uint8")


def _not_too_close_to_zero(x):
    f = np.vectorize(lambda item: item + (_one if np.isclose(item, 0) else _zero))
    return f(x)


# abs
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="abs"),
)
def test_abs(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="abs",
        x=np.asarray(x, dtype=input_dtype),
    )


# acosh
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="acosh"),
)
def test_acosh(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="acosh",
        x=np.asarray(x, dtype=input_dtype),
    )


# acos
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="acos"),
)
def test_acos(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="acos",
        x=np.asarray(x, dtype=input_dtype),
    )


# add
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="add"),
)
def test_add(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="add",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# asin
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="asin"),
)
def test_asin(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="asin",
        x=np.asarray(x, dtype=input_dtype),
    )


# asinh
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="asinh"),
)
def test_asinh(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="asinh",
        x=np.asarray(x, dtype=input_dtype),
    )


# atan
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="atan"),
)
def test_atan(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="atan",
        x=np.asarray(x, dtype=input_dtype),
    )


# atan2
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="atan2"),
)
def test_atan2(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    x1 = np.asarray(x[0], dtype=input_dtype[0])
    x2 = np.asarray(x[1], dtype=input_dtype[1])

    x1 = _not_too_close_to_zero(x1)
    x2 = _not_too_close_to_zero(x2)

    assume(not (np.any(np.isclose(x1, 0)) or np.any(np.isclose(x2, 0))))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="atan2",
        x1=x1,
        x2=x2,
    )


# atanh
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="atanh"),
)
def test_atanh(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="atanh",
        x=np.asarray(x, dtype=input_dtype),
    )


# bitwise_and
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes + ("bool",), num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_and"),
)
def test_bitwise_and(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="bitwise_and",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# bitwise_left_shift
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes,
        num_arrays=2,
        shared_dtype=True,
        large_value_safety_factor=0.9,
        small_value_safety_factor=0.9,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_left_shift"),
)
def test_bitwise_left_shift(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    # make sure x2 is not negative
    if "int" in input_dtype[0] and "int" in input_dtype[1]:
        x[1] = np.abs(x[1])

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="bitwise_left_shift",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# bitwise_invert
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes + ("bool",)
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_invert"),
)
def test_bitwise_invert(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="bitwise_invert",
        x=np.asarray(x, dtype=input_dtype),
    )


# bitwise_or
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes + ("bool",), num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_or"),
)
def test_bitwise_or(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="bitwise_or",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# bitwise_right_shift
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes,
        num_arrays=2,
        shared_dtype=True,
        large_value_safety_factor=0.9,
        small_value_safety_factor=0.9,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_right_shift"),
)
def test_bitwise_right_shift(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    # make sure x2 is not negative
    if "int" in input_dtype[0] and "int" in input_dtype[1]:
        x[1] = np.abs(x[1])

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="bitwise_right_shift",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# bitwise_xor
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes + ("bool",), num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_xor"),
)
def test_bitwise_xor(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="bitwise_xor",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# ceil
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="ceil"),
)
def test_ceil(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="ceil",
        x=np.asarray(x, dtype=input_dtype),
    )


# cos
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="cos"),
)
def test_cos(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cos",
        x=np.asarray(x, dtype=input_dtype),
    )


# cosh
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="cosh"),
)
def test_cosh(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cosh",
        x=np.asarray(x, dtype=input_dtype),
    )


# divide
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="divide"),
)
def test_divide(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    x1 = np.asarray(x[0], dtype=input_dtype[0])
    x2 = np.asarray(x[1], dtype=input_dtype[1])

    # prevent too close to zero
    assume(not np.any(np.isclose(x2, 0)))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="divide",
        x1=x1,
        x2=x2,
    )


# equal
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="equal"),
)
def test_equal(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="equal",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# exp
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="exp"),
)
def test_exp(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="exp",
        x=np.asarray(x, dtype=input_dtype),
    )


# expm1
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="expm1"),
)
def test_expm1(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="expm1",
        x=np.asarray(x, dtype=input_dtype),
    )


# floor
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="floor"),
)
def test_floor(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    x = np.asarray(x, dtype=input_dtype)
    assume(not np.any(np.isclose(x, 0)))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="floor",
        x=x,
    )


@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=2,
        allow_inf=False,
        large_value_safety_factor=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="floor_divide"),
)
def test_floor_divide(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    x1 = np.asarray(x[0], dtype=input_dtype[0])
    x2 = np.asarray(x[1], dtype=input_dtype[1])

    # Make sure it's not dividing value too close to zero
    assume(not np.any(np.isclose(x2, 0)))

    # Absolute tolerance is 1,
    # due to flooring can cause absolute error of 1 due to precision
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="floor_divide",
        x1=x1,
        x2=x2,
        atol_=1,
    )


# greater
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="greater"),
)
def test_greater(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    x1 = np.asarray(x[0], dtype=input_dtype[0])
    x2 = np.asarray(x[1], dtype=input_dtype[1])

    # make sure they're not too close together
    assume(not (np.any(np.isclose(x1, x2)) or np.any(np.isclose(x2, x1))))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="greater",
        x1=x1,
        x2=x2,
    )


# greater_equal
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="greater_equal"),
)
def test_greater_equal(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    x1 = np.asarray(x[0], dtype=input_dtype[0])
    x2 = np.asarray(x[1], dtype=input_dtype[1])

    # make sure they're not too close together
    assume(not (np.any(np.isclose(x1, x2)) or np.any(np.isclose(x2, x1))))
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="greater_equal",
        x1=x1,
        x2=x2,
    )


# isfinite
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="isfinite"),
)
def test_isfinite(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="isfinite",
        x=np.asarray(x, dtype=input_dtype),
    )


# isinf
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="isinf"),
)
def test_isinf(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="isinf",
        x=np.asarray(x, dtype=input_dtype),
    )


# isnan
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="isnan"),
)
def test_isnan(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="isnan",
        x=np.asarray(x, dtype=input_dtype),
    )


# less
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=2,
        min_num_dims=1,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="less"),
)
def test_less(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    x1 = np.asarray(x[0], dtype=input_dtype[0])
    x2 = np.asarray(x[1], dtype=input_dtype[1])

    # make sure they're not too close together
    assume(not (np.any(np.isclose(x1, x2)) or np.any(np.isclose(x2, x1))))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="less",
        x1=x1,
        x2=x2,
    )


# less_equal
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="less_equal"),
)
def test_less_equal(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    x1 = np.asarray(x[0], dtype=input_dtype[0])
    x2 = np.asarray(x[1], dtype=input_dtype[1])

    # make sure they're not too close together
    assume(not (np.any(np.isclose(x1, x2)) or np.any(np.isclose(x2, x1))))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="less_equal",
        x1=x1,
        x2=x2,
    )


# log
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="log"),
)
def test_log(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    # avoid logging values too close to zero
    assume(not np.any(np.isclose(x, 0)))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="log",
        x=np.asarray(x, dtype=input_dtype),
    )


# log1p
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="log1p"),
)
def test_log1p(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    # avoid logging values too close to zero
    assume(not np.any(np.isclose(x, 0)))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="log1p",
        x=np.asarray(x, dtype=input_dtype),
    )


# log2
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="log2"),
)
def test_log2(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    # avoid logging values too close to zero
    assume(not np.any(np.isclose(x, 0)))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="log2",
        x=np.asarray(x, dtype=input_dtype),
    )


# log10
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="log10"),
)
def test_log10(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    # avoid logging values too close to zero
    assume(not np.any(np.isclose(x, 0)))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="log10",
        x=np.asarray(x, dtype=input_dtype),
    )


# logaddexp
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="logaddexp"),
)
def test_logaddexp(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="logaddexp",
        rtol_=1e-2,
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# logical_and
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=("bool",), num_arrays=2),
    num_positional_args=helpers.num_positional_args(fn_name="logical_and"),
)
def test_logical_and(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="logical_and",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# logical_not
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=("bool",)),
    num_positional_args=helpers.num_positional_args(fn_name="logical_not"),
)
def test_logical_not(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="logical_not",
        x=np.asarray(x, dtype=input_dtype),
    )


# logical_or
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=("bool",), num_arrays=2),
    num_positional_args=helpers.num_positional_args(fn_name="logical_or"),
)
def test_logical_or(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="logical_or",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# logical_xor
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=("bool",), num_arrays=2),
    num_positional_args=helpers.num_positional_args(fn_name="logical_xor"),
)
def test_logical_xor(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="logical_xor",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# multiply
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="multiply"),
)
def test_multiply(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="multiply",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# negative
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="negative"),
)
def test_negative(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="negative",
        x=np.asarray(x, dtype=input_dtype),
    )


# not_equal
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="not_equal"),
)
def test_not_equal(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="not_equal",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# positive
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="positive"),
)
def test_positive(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="positive",
        x=np.asarray(x, dtype=input_dtype),
    )


# pow
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="pow"),
)
def test_pow(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    # input_dtype, x = (['int16', 'int8'], [[2], [64]])

    # Make sure x2 isn't a float when x1 is integer
    assume(
        not (ivy.is_int_dtype(input_dtype[0] and ivy.is_float_dtype(input_dtype[1])))
    )

    # Make sure x2 is non-negative when both is integer
    if ivy.is_int_dtype(input_dtype[1]) and ivy.is_int_dtype(input_dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])

    # Makesure it doesn't overflow
    nt = np.promote_types(input_dtype[0], input_dtype[1])
    size = nt.itemsize * 8 - 1
    assume(np.all(np.log2(x[0]) * x[1] <= size))

    x1 = np.asarray(x[0], dtype=input_dtype[0])
    x2 = np.asarray(x[1], dtype=input_dtype[1])

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="pow",
        x1=x1,
        x2=x2,
    )


# remainder
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2, allow_inf=False
    ),
    num_positional_args=helpers.num_positional_args(fn_name="remainder"),
)
def test_remainder(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    x1 = np.asarray(x[0], dtype=input_dtype[0])
    x2 = np.asarray(x[1], dtype=input_dtype[1])

    # Make sure values is not too close to zero
    assume(not np.any(np.isclose(x1, 0)))
    assume(not np.any(np.isclose(x2, 0)))

    native_array = [native_array, native_array]
    container = [container, container]

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=[as_variable, False],
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="remainder",
        x1=x1,
        x2=x2,
    )


# round
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="round"),
)
def test_round(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="round",
        x=np.asarray(x, dtype=input_dtype),
    )


# sign
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="sign"),
)
def test_sign(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    x = np.asarray(x, dtype=input_dtype)
    assume(not np.any(np.isclose(x, 0)))

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sign",
        x=np.asarray(x, dtype=input_dtype),
    )


# sin
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="sin"),
)
def test_sin(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sin",
        x=np.asarray(x, dtype=input_dtype),
    )


# sinh
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="sinh"),
)
def test_sinh(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sinh",
        x=np.asarray(x, dtype=input_dtype),
    )


# square
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="square"),
)
def test_square(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="square",
        x=np.asarray(x, dtype=input_dtype),
    )


# sqrt
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, allow_inf=False
    ),
    num_positional_args=helpers.num_positional_args(fn_name="sqrt"),
)
def test_sqrt(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sqrt",
        x=np.asarray(x, dtype=input_dtype),
    )


# subtract
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="subtract"),
)
def test_subtract(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="subtract",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# tan
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="tan"),
)
def test_tan(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="tan",
        x=np.asarray(x, dtype=input_dtype),
    )


# tanh
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="tanh"),
)
def test_tanh(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="tanh",
        x=np.asarray(x, dtype=input_dtype),
    )


# trunc
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="trunc"),
)
def test_trunc(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="trunc",
        x=np.asarray(x, dtype=input_dtype),
    )


# Extra #
# ------#


# erf
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="erf"),
)
def test_erf(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="erf",
        x=np.asarray(x, dtype=input_dtype),
    )


# minimum
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="minimum"),
)
def test_minimum(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    assume(
        not (
            (
                (isinstance(x[0], Number) or isinstance(x[1], Number))
                and as_variable is True
                and fw == "mxnet"
            )
        )
    )

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="minimum",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# maximum
@handle_cmd_line_args()
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="maximum"),
)
def test_maximum(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    assume(
        not (
            (isinstance(x[0], Number) or isinstance(x[1], Number))
            and as_variable is True
            and fw == "mxnet"
        )
    )

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="maximum",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )
