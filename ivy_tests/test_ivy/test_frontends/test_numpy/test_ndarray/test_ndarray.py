# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


# argmax
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_numpy_ndarray_argmax(
    dtype_and_x,
    as_variable,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtype_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "value": x[1],
        },
        fw=fw,
        frontend="numpy",
        class_name="ndarray",
        method_name="argmax",
    )


# reshape
@st.composite
def dtypes_x_reshape(draw):
    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=10,
            ),
        )
    )
    shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
    return dtypes, x, shape


@handle_cmd_line_args
@given(
    dtypes_x_shape=dtypes_x_reshape(),
)
def test_numpy_ndarray_reshape(
    dtypes_x_shape,
    as_variable,
    native_array,
):
    input_dtype, x, shape = dtypes_x_shape
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "shape": shape,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="reshape",
    )


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_numpy_ndarray_add(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "value": x[1],
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="add",
    )


# transpose
@handle_cmd_line_args
@given(
    array_and_axes=np_frontend_helpers._array_and_axes_permute_helper(
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.transpose"
    ),
)
def test_numpy_ndarray_transpose(
    array_and_axes,
    as_variable,
    num_positional_args,
    native_array,
):
    array, dtype, axes = array_and_axes
    helpers.test_frontend_method(
        input_dtypes_init=dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=num_positional_args,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": np.array(array),
        },
        input_dtypes_method=dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "axes": axes,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="transpose",
    )


# any
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=True,
    ),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.any"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_ndarray_any(
    dtype_x_axis,
    keepdims,
    where,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x, axis = dtype_x_axis
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=num_positional_args,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "axis": axis,
            "out": None,
            "keepdims": keepdims,
            "where": where,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="any",
    )


# all
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=True,
    ),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.all"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_ndarray_all(
    dtype_x_axis,
    keepdims,
    where,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x, axis = dtype_x_axis
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=num_positional_args,
        num_positional_args_method=num_positional_args,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        all_as_kwargs_np_method={
            "axis": axis,
            "out": None,
            "keepdims": keepdims,
            "where": where,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="all",
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.argsort"
    ),
)
def test_numpy_instance_argsort(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_instance_method(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        frontend_class=np.ndarray,
        fn_tree="ndarray.argsort",
        x=x[0],
        axis=axis,
    )
