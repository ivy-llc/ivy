# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
from ivy_tests.test_ivy.helpers import handle_frontend_method


CLASS_TREE = "ivy.functional.frontends.numpy.matrix"


def _to_string_matrix(num_matrix):
    str_matrix = ''
    for i, row in enumerate(num_matrix):
        for j, elem in enumerate(row):
            str_matrix += str(elem)
            if j < num_matrix.shape[1] - 1:
                str_matrix += ' '
            elif i < num_matrix.shape[0] - 1:
                str_matrix += '; '
    return str_matrix


def _get_x_matrix(x, to_str):
    if to_str:
        x = _to_string_matrix(x[0])
    else:
        x = x[0]
    return x


# argmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.matrix",
    method_name="argmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    to_str=st.booleans(),
)
def test_numpy_matrix_argmax(
    dtype_x_axis,
    to_str,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    frontend_method_data,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    x = _get_x_matrix(x, to_str)
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x,
            "dtype": input_dtype[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_native_array_flags=native_array,
        method_num_positional_args=method_num_positional_args,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
    )


# any
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.matrix",
    method_name="any",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    to_str=st.booleans(),
)
def test_numpy_matrix_any(
    dtype_x_axis,
    to_str,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    frontend_method_data,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    x = _get_x_matrix(x, to_str)
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x,
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_native_array_flags=native_array,
        method_num_positional_args=method_num_positional_args,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
    )
