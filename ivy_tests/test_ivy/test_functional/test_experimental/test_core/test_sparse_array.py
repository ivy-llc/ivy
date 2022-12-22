# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method
from ivy_tests.test_ivy.helpers import test_parameter_flags as pf

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
            exclude_min=False,
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
            exclude_min=False,
        )
    )
    indices = draw(
        helpers.array_values(
            dtype="int64",
            shape=(dim1 - 1,),
            min_value=0,
            max_value=num_elem,
            exclude_min=False,
        )
    )
    crow_indices = [0] + sorted(indices) + [num_elem]
    shape = (dim1, dim2)
    return crow_indices, col_indices, value_dtype, values, shape


@st.composite
def _sparse_csc_indices_values_shape(draw):
    num_elem = draw(helpers.ints(min_value=2, max_value=8))
    dim1 = draw(helpers.ints(min_value=5, max_value=10))
    dim2 = draw(helpers.ints(min_value=2, max_value=5))
    value_dtype = draw(helpers.get_dtypes("numeric", full=False))[0]
    values = draw(helpers.array_values(dtype=value_dtype, shape=(num_elem,)))
    row_indices = draw(
        helpers.array_values(
            dtype="int64",
            shape=(num_elem,),
            min_value=0,
            max_value=dim1,
            exclude_min=False,
        )
    )
    indices = draw(
        helpers.array_values(
            dtype="int64",
            shape=(dim2 - 1,),
            min_value=0,
            max_value=num_elem,
            exclude_min=False,
        )
    )
    ccol_indices = [0] + sorted(indices) + [num_elem]
    shape = (dim1, dim2)
    return ccol_indices, row_indices, value_dtype, values, shape


# coo - to_dense_array
@handle_method(
    method_tree="SparseArray.to_dense_array",
    sparse_data=_sparse_coo_indices_values_shape(),
)
def test_sparse_coo(
    sparse_data,
    init_as_variable_flags: pf.AsVariableFlags,
    init_native_array_flags: pf.NativeArrayFlags,
    class_name,
    method_name,
    ground_truth_backend,
):
    coo_ind, val_dtype, val, shp = sparse_data
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_input_dtypes=["int64", val_dtype],
        init_as_variable_flags=init_as_variable_flags,
        init_num_positional_args=0,
        init_native_array_flags=init_native_array_flags,
        init_all_as_kwargs_np={
            "coo_indices": coo_ind,
            "values": val,
            "dense_shape": shp,
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_container_flags=[False],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


# csr - to_dense_array
@handle_method(
    method_tree="SparseArray.to_dense_array",
    sparse_data=_sparse_csr_indices_values_shape(),
)
def test_sparse_csr(
    sparse_data,
    init_as_variable_flags: pf.AsVariableFlags,
    init_native_array_flags: pf.NativeArrayFlags,
    class_name,
    method_name,
    ground_truth_backend,
):
    crow_indices, col_indices, value_dtype, values, shape = sparse_data
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_input_dtypes=["int64", "int64", value_dtype],
        init_as_variable_flags=init_as_variable_flags,
        init_num_positional_args=0,
        init_native_array_flags=init_native_array_flags,
        init_all_as_kwargs_np={
            "csr_crow_indices": crow_indices,
            "csr_col_indices": col_indices,
            "values": values,
            "dense_shape": shape,
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_container_flags=[False],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


# csc - to_dense_array
@handle_method(
    method_tree="SparseArray.to_dense_array",
    sparse_data=_sparse_csc_indices_values_shape(),
)
def test_sparse_csc(
    sparse_data,
    init_as_variable_flags: pf.AsVariableFlags,
    init_native_array_flags: pf.NativeArrayFlags,
    class_name,
    method_name,
    ground_truth_backend,
):
    ccol_indices, row_indices, value_dtype, values, shape = sparse_data
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_input_dtypes=["int64", "int64", value_dtype],
        init_as_variable_flags=init_as_variable_flags,
        init_num_positional_args=0,
        init_native_array_flags=init_native_array_flags,
        init_all_as_kwargs_np={
            "csc_ccol_indices": ccol_indices,
            "csc_row_indices": row_indices,
            "values": values,
            "dense_shape": shape,
        },
        method_input_dtypes=[],
        method_as_variable_flags=[],
        method_num_positional_args=0,
        method_native_array_flags=[],
        method_container_flags=[False],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )
