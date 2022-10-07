# global
from hypothesis import given, strategies as st

# local
import numpy as np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Helpers #
# ------- #


@st.composite
def _sparse_coo_indices_values_shape(draw):
    num_elem = draw(helpers.ints(min_value=2, max_value=8))
    dim1 = draw(helpers.ints(min_value=2, max_value=5))
    dim2 = draw(helpers.ints(min_value=5, max_value=10))
    value_dtype = draw(helpers.get_dtypes("numeric", full=False))[0]
    coo_indices = draw(
        helpers.array_values(
            dtype="int64",
            shape=(2, num_elem),
            min_value=0,
            max_value=dim1,
        )
    )
    values = draw(helpers.array_values(dtype=value_dtype, shape=(num_elem,)))
    shape = (dim1, dim2)
    return coo_indices, value_dtype, values, shape


@st.composite
def _sparse_csr_indices_values_shape(draw):
    num_elem = draw(helpers.ints(min_value=2, max_value=8))
    dim1 = draw(helpers.ints(min_value=2, max_value=5))
    dim2 = draw(helpers.ints(min_value=5, max_value=10))
    value_dtype = draw(helpers.get_dtypes("numeric", full=False))[0]
    values = draw(helpers.array_values(dtype=value_dtype, shape=(num_elem,)))
    col_indices = draw(
        helpers.array_values(
            dtype="int64",
            shape=(num_elem,),
            min_value=0,
            max_value=dim2,
        )
    )
    indices = draw(
        helpers.array_values(
            dtype="int64",
            shape=(dim1 - 1,),
            min_value=0,
            max_value=num_elem,
        )
    )
    crow_indices = [0] + sorted(indices) + [num_elem]
    shape = (dim1, dim2)
    return crow_indices, col_indices, value_dtype, values, shape


# coo - to_dense_array
@handle_cmd_line_args
@given(sparse_data=_sparse_coo_indices_values_shape())
def test_sparse_coo(
    sparse_data,
    as_variable,
    with_out,
    native_array,
    fw,
):
    coo_ind, val_dtype, val, shp = sparse_data
    helpers.test_method(
        input_dtypes_init=["int64", val_dtype],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "coo_indices": np.array(coo_ind, dtype="int64"),
            "values": np.array(val, dtype=val_dtype),
            "dense_shape": shp,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=False,
        all_as_kwargs_np_method={},
        class_name="SparseArray",
        method_name="to_dense_array",
    )


# csr - to_dense_array
@handle_cmd_line_args
@given(sparse_data=_sparse_csr_indices_values_shape())
def test_sparse_csr(
    sparse_data,
    as_variable,
    with_out,
    native_array,
    fw,
):
    crow_indices, col_indices, value_dtype, values, shape = sparse_data
    helpers.test_method(
        input_dtypes_init=["int64", "int64", value_dtype],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "csr_crow_indices": np.array(crow_indices, dtype="int64"),
            "csr_col_indices": np.array(col_indices, dtype="int64"),
            "values": np.array(values, dtype=value_dtype),
            "dense_shape": shape,
        },
        input_dtypes_method=[],
        as_variable_flags_method=[],
        num_positional_args_method=0,
        native_array_flags_method=[],
        container_flags_method=False,
        all_as_kwargs_np_method={},
        class_name="SparseArray",
        method_name="to_dense_array",
    )


# sinc
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="sinc"),
)
def test_sinc(
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
        fn_name="sinc",
        x=np.asarray(x, dtype=input_dtype),
    )


# vorbis_window
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=1, max_num_dims=1
    ),
    dtype=helpers.get_dtypes("float", full=False),
    num_positional_args=helpers.num_positional_args(fn_name="vorbis_window"),
)
def test_vorbis_window(
    dtype_and_x,
    dtype,
    with_out,
    as_variable,
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
        fn_name="vorbis_window",
        x=x[0],
        dtype=dtype,
    )
