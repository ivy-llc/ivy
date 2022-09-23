# global
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# argmax
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
)
def test_numpy_matrix_argmax(
    dtype_x_axis,
    as_variable,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_method(
        input_dtypes_init=[input_dtype],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={
            "data": x,
            "dtype": input_dtype,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "axis": axis,
        },
        fw=fw,
        frontend="numpy",
        class_name="matrix",
        method_name="argmax",
    )


# any
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
)
def test_numpy_matrix_any(
    dtype_x_axis,
    as_variable,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_method(
        input_dtypes_init=[input_dtype],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={
            "data": x,
            "dtype": input_dtype,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "axis": axis,
        },
        fw=fw,
        frontend="numpy",
        class_name="matrix",
        method_name="any",
    )
