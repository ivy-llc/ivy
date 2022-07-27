"""Collection of tests for elementwise functions."""

# global
import numpy as np
from hypothesis import given, assume, strategies as st
from numbers import Number

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy.functional.backends.numpy as ivy_np


# abs
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="abs"),
    data=st.data(),
)
@handle_cmd_line_args
def test_abs(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="acosh"),
    data=st.data(),
)
@handle_cmd_line_args
def test_acosh(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="acos"),
    data=st.data(),
)
@handle_cmd_line_args
def test_acos(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="add"),
)
@handle_cmd_line_args
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="asin"),
    data=st.data(),
)
@handle_cmd_line_args
def test_asin(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="asinh"),
    data=st.data(),
)
@handle_cmd_line_args
def test_asinh(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="atan"),
    data=st.data(),
)
@handle_cmd_line_args
def test_atan(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="atan2"),
    data=st.data(),
)
@handle_cmd_line_args
def test_atan2(
    *,
    data,
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
        fn_name="atan2",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# atanh
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="atanh"),
    data=st.data(),
)
@handle_cmd_line_args
def test_atanh(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes + ("bool",), num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_and"),
    data=st.data(),
)
@handle_cmd_line_args
def test_bitwise_and(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes, num_arrays=2, shared_dtype=True
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_left_shift"),
    data=st.data(),
)
@handle_cmd_line_args
def test_bitwise_left_shift(
    *,
    data,
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
        fn_name="bitwise_left_shift",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# bitwise_invert
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes + ("bool",)
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_invert"),
    data=st.data(),
)
@handle_cmd_line_args
def test_bitwise_invert(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes + ("bool",), num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_or"),
    data=st.data(),
)
@handle_cmd_line_args
def test_bitwise_or(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes, num_arrays=2, shared_dtype=True
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_right_shift"),
    data=st.data(),
)
@handle_cmd_line_args
def test_bitwise_right_shift(
    *,
    data,
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
        fn_name="bitwise_right_shift",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# bitwise_xor
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.all_int_dtypes + ("bool",), num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="bitwise_xor"),
    data=st.data(),
)
@handle_cmd_line_args
def test_bitwise_xor(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="ceil"),
    data=st.data(),
)
@handle_cmd_line_args
def test_ceil(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="cos"),
    data=st.data(),
)
@handle_cmd_line_args
def test_cos(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="cosh"),
    data=st.data(),
)
@handle_cmd_line_args
def test_cosh(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="divide"),
    data=st.data(),
)
@handle_cmd_line_args
def test_divide(
    *,
    data,
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
    # prevent division by 0
    assume(np.all(x2 != 0))

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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="equal"),
    data=st.data(),
)
@handle_cmd_line_args
def test_equal(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="exp"),
    data=st.data(),
)
@handle_cmd_line_args
def test_exp(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="expm1"),
    data=st.data(),
)
@handle_cmd_line_args
def test_expm1(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="floor"),
    data=st.data(),
)
@handle_cmd_line_args
def test_floor(
    *,
    data,
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
        fn_name="floor",
        x=np.asarray(x, dtype=input_dtype),
    )


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=2,
        allow_inf=False,
        safety_factor=0.5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="floor_divide"),
    data=st.data(),
)
@handle_cmd_line_args
def test_floor_divide(
    *,
    data,
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
    x1 = (np.asarray(x[0], dtype=input_dtype[0]),)
    x2 = (np.asarray(x[1], dtype=input_dtype[1]),)
    assume(np.all(x2[0] != 0))

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
        x1=x1[0],
        x2=x2[0],
    )


# greater
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="greater"),
    data=st.data(),
)
@handle_cmd_line_args
def test_greater(
    *,
    data,
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
        fn_name="greater",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# greater_equal
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="greater_equal"),
    data=st.data(),
)
@handle_cmd_line_args
def test_greater_equal(
    *,
    data,
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
        fn_name="greater_equal",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# isfinite
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="isfinite"),
    data=st.data(),
)
@handle_cmd_line_args
def test_isfinite(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="isinf"),
    data=st.data(),
)
@handle_cmd_line_args
def test_isinf(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="isnan"),
    data=st.data(),
)
@handle_cmd_line_args
def test_isnan(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="less"),
    data=st.data(),
)
@handle_cmd_line_args
def test_less(
    *,
    data,
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
        fn_name="less",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# less_equal
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="less_equal"),
    data=st.data(),
)
@handle_cmd_line_args
def test_less_equal(
    *,
    data,
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
        fn_name="less_equal",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# log
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="log"),
    data=st.data(),
)
@handle_cmd_line_args
def test_log(
    *,
    data,
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
        fn_name="log",
        x=np.asarray(x, dtype=input_dtype),
    )


# log1p
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="log1p"),
    data=st.data(),
)
@handle_cmd_line_args
def test_log1p(
    *,
    data,
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
        fn_name="log1p",
        x=np.asarray(x, dtype=input_dtype),
    )


# log2
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="log2"),
    data=st.data(),
)
@handle_cmd_line_args
def test_log2(
    *,
    data,
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
        fn_name="log2",
        x=np.asarray(x, dtype=input_dtype),
    )


# log10
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="log10"),
    data=st.data(),
)
@handle_cmd_line_args
def test_log10(
    *,
    data,
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
        fn_name="log10",
        x=np.asarray(x, dtype=input_dtype),
    )


# logaddexp
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="logaddexp"),
    data=st.data(),
)
@handle_cmd_line_args
def test_logaddexp(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=("bool",), num_arrays=2),
    num_positional_args=helpers.num_positional_args(fn_name="logical_and"),
    data=st.data(),
)
@handle_cmd_line_args
def test_logical_and(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=("bool",)),
    num_positional_args=helpers.num_positional_args(fn_name="logical_not"),
    data=st.data(),
)
@handle_cmd_line_args
def test_logical_not(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=("bool",), num_arrays=2),
    num_positional_args=helpers.num_positional_args(fn_name="logical_or"),
    data=st.data(),
)
@handle_cmd_line_args
def test_logical_or(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=("bool",), num_arrays=2),
    num_positional_args=helpers.num_positional_args(fn_name="logical_xor"),
    data=st.data(),
)
@handle_cmd_line_args
def test_logical_xor(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="multiply"),
    data=st.data(),
)
@handle_cmd_line_args
def test_multiply(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="negative"),
    data=st.data(),
)
@handle_cmd_line_args
def test_negative(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="not_equal"),
    data=st.data(),
)
@handle_cmd_line_args
def test_not_equal(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="positive"),
    data=st.data(),
)
@handle_cmd_line_args
def test_positive(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="pow"),
    data=st.data(),
)
@handle_cmd_line_args
def test_pow(
    *,
    data,
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
    assume(
        not (
            np.any(x2 < 0)
            and ivy.is_int_dtype(input_dtype[1])
            and ivy.is_int_dtype(input_dtype[0])
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
        fn_name="pow",
        x1=x1,
        x2=x2,
    )


# remainder
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2, allow_inf=False
    ),
    num_positional_args=helpers.num_positional_args(fn_name="remainder"),
    data=st.data(),
)
@handle_cmd_line_args
def test_remainder(
    *,
    data,
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
    assume(not np.any(x2 == 0))

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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="round"),
    data=st.data(),
)
@handle_cmd_line_args
def test_round(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="sign"),
    data=st.data(),
)
@handle_cmd_line_args
def test_sign(
    *,
    data,
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
        fn_name="sign",
        x=np.asarray(x, dtype=input_dtype),
    )


# sin
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="sin"),
    data=st.data(),
)
@handle_cmd_line_args
def test_sin(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="sinh"),
    data=st.data(),
)
@handle_cmd_line_args
def test_sinh(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="square"),
    data=st.data(),
)
@handle_cmd_line_args
def test_square(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, allow_inf=False
    ),
    num_positional_args=helpers.num_positional_args(fn_name="sqrt"),
    data=st.data(),
)
@handle_cmd_line_args
def test_sqrt(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="subtract"),
    data=st.data(),
)
@handle_cmd_line_args
def test_subtract(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="tan"),
    data=st.data(),
)
@handle_cmd_line_args
def test_tan(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="tanh"),
    data=st.data(),
)
@handle_cmd_line_args
def test_tanh(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="trunc"),
    data=st.data(),
)
@handle_cmd_line_args
def test_trunc(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="erf"),
    data=st.data(),
)
@handle_cmd_line_args
def test_erf(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="minimum"),
    data=st.data(),
)
@handle_cmd_line_args
def test_minimum(
    *,
    data,
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
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="maximum"),
    data=st.data(),
)
@handle_cmd_line_args
def test_maximum(
    *,
    data,
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
