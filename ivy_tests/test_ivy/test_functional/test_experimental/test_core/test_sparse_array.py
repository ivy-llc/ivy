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
        init_input_dtypes=["int64", val_dtype],
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "coo_indices": np.array(coo_ind, dtype="int64"),
            "values": np.array(val, dtype=val_dtype),
            "dense_shape": shp,
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_container_flags=False,
        method_all_as_kwargs_np={},
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
        init_input_dtypes=["int64", "int64", value_dtype],
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "csr_crow_indices": np.array(crow_indices, dtype="int64"),
            "csr_col_indices": np.array(col_indices, dtype="int64"),
            "values": np.array(values, dtype=value_dtype),
            "dense_shape": shape,
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_container_flags=False,
        method_all_as_kwargs_np={},
        class_name="SparseArray",
        method_name="to_dense_array",
    )
