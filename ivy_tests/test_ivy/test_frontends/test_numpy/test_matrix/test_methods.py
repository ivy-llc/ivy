# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method


# A
@handle_frontend_method(
    method_tree="numpy.matrix.A",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
)
def test_numpy_matrix_A(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="matrix",
        method_name="A",
    )


# A1
@handle_frontend_method(
    method_tree="numpy.matrix.A1",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
)
def test_numpy_matrix_A1(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="matrix",
        method_name="A1",
    )


# I
@handle_frontend_method(
    method_tree="numpy.matrix.I",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
)
def test_numpy_matrix_I(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="matrix",
        method_name="I",
    )


# T
@handle_frontend_method(
    method_tree="numpy.matrix.T",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
)
def test_numpy_matrix_T(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="matrix",
        method_name="T",
    )


# data
@handle_frontend_method(
    method_tree="numpy.matrix.data",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
)
def test_numpy_matrix_data(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="matrix",
        method_name="data",
    )


# dtype
@handle_frontend_method(
    method_tree="numpy.matrix.dtype",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
)
def test_numpy_matrix_dtype(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="matrix",
        method_name="dtype",
    )


# ndim
@handle_frontend_method(
    method_tree="numpy.matrix.ndim",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
)
def test_numpy_matrix_ndim(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="matrix",
        method_name="ndim",
    )


# shape
@handle_frontend_method(
    method_tree="numpy.matrix.shape",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
)
def test_numpy_matrix_shape(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="matrix",
        method_name="shape",
    )


# size
@handle_frontend_method(
    method_tree="numpy.matrix.size",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
)
def test_numpy_matrix_size(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="matrix",
        method_name="size",
    )


# argmax
@handle_frontend_method(
    method_tree="numpy.matrix.argmax",
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
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend="numpy",
        class_="matrix",
        method_name="argmax",
    )


# any
@handle_frontend_method(
    method_tree="numpy.matrix.any",
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
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend="numpy",
        class_="matrix",
        method_name="any",
    )
