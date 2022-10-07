# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(min_value=-1e04, max_value=1e04, allow_infinity=False),
)
def test_torch_instance_add(
    dtype_and_x,
    alpha,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
            "alpha": alpha,
        },
        frontend="torch",
        class_name="tensor",
        method_name="add",
    )


@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    shape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(), key="value_shape")
    ),
)
def test_torch_instance_reshape(
    dtype_x,
    shape,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "shape": shape,
        },
        frontend="torch",
        class_name="tensor",
        method_name="reshape",
    )


# sin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sin(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="sin",
    )


# sin_
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sin_(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="sin_",
    )
