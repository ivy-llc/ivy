# global
from hypothesis import assume, given, strategies as st
import math
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


_zero = np.asarray(0, dtype="uint8")
_one = np.asarray(1, dtype="uint8")


def _not_too_close_to_zero(x):
    f = np.vectorize(lambda item: item + (_one if np.isclose(item, 0) else _zero))
    return f(x)


@st.composite
def _pow_helper(draw, available_dtypes=None):
    if available_dtypes is None:
        available_dtypes = helpers.get_dtypes("numeric")
    dtype1, x1 = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            small_abs_safety_factor=4,
            large_abs_safety_factor=4,
        )
    )
    dtype1 = dtype1[0]

    def cast_filter(dtype1_x1_dtype2):
        dtype1, _, dtype2 = dtype1_x1_dtype2
        if (ivy.as_ivy_dtype(dtype1), ivy.as_ivy_dtype(dtype2)) in ivy.promotion_table:
            return True
        return False

    dtype1, x1, dtype2 = draw(
        helpers.get_castable_dtype(draw(available_dtypes), dtype1, x1).filter(
            cast_filter
        )
    )
    if ivy.is_int_dtype(dtype2):
        max_val = ivy.iinfo(dtype2).max
    else:
        max_val = ivy.finfo(dtype2).max
    max_x1 = np.max(np.abs(x1[0]))
    if max_x1 in [0, 1]:
        max_value = None
    else:
        max_value = int(math.log(max_val) / math.log(max_x1))
        if abs(max_value) > abs(max_val) / 40 or max_value < 0:
            max_value = None
    dtype2, x2 = draw(
        helpers.dtype_and_values(
            small_abs_safety_factor=12,
            large_abs_safety_factor=12,
            safety_factor_scale="log",
            max_value=max_value,
            dtype=[dtype2],
        )
    )
    dtype2 = dtype2[0]
    if "int" in dtype2:
        x2 = ivy.nested_map(x2[0], lambda x: abs(x), include_derived={list: True})
    return [dtype1, dtype2], [x1, x2]


# __pos__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="ivy.Array.__pos__"),
)
def test_array__pos__(
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        input_dtypes_init=dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        container_flags_method=False,
        all_as_kwargs_np_method={},
        fw=fw,
        class_name="Array",
        method_name="__pos__",
    )


# __neg__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="ivy.Array.__neg__"),
)
def test_array__neg__(
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        input_dtypes_init=["int64", dtype],
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        container_flags_method=False,
        all_as_kwargs_np_method={},
        fw=fw,
        class_name="Array",
        method_name="__neg__",
    )


# __pow__
@handle_cmd_line_args
@given(
    dtype_and_x=_pow_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="ivy.Array.__pow__"),
)
def test_array__pow__(
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # check if power isn't a float when x1 is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]):
        x[1] = np.abs(x[1])
    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    helpers.test_method(
        input_dtypes_init=["int64", dtype],
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        container_flags_method=False,
        all_as_kwargs_np_method={"power": x[1]},
        fw=fw,
        class_name="Array",
        method_name="__pow__",
    )


# __rpow__
@handle_cmd_line_args
@given(
    dtype_and_x=_pow_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="ivy.Array.__rpow__"),
)
def test_array__rpow__(
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))

    # check if power isn't a float when x1 is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))

    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]) and ivy.is_int_dtype(dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    helpers.test_method(
        input_dtypes_init=["int64", dtype],
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[1],
        },
        input_dtypes_method=dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        container_flags_method=False,
        all_as_kwargs_np_method={
            "power": x[0],
        },
        fw=fw,
        class_name="Array",
        method_name="__rpow__",
    )


# __ipow__
@handle_cmd_line_args
@given(
    dtype_and_x=_pow_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="ivy.Array.__ipow__"),
)
def test_array__ipow__(
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # check if power isn't a float when data is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]) and ivy.is_int_dtype(dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    helpers.test_method(
        input_dtypes_init=["int64", dtype],
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        container_flags_method=False,
        all_as_kwargs_np_method={"power": x[1]},
        fw=fw,
        class_name="Array",
        method_name="__ipow__",
    )


# __add__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    num_positional_args=helpers.num_positional_args(fn_name="ivy.Array.__add__"),
)
def test_array__add__(
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        input_dtypes_init=dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        container_flags_method=False,
        all_as_kwargs_np_method={
            "other": x[1],
        },
        fw=fw,
        class_name="Array",
        method_name="__add__",
    )
