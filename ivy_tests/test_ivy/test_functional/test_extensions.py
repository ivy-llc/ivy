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
    value_dtype = draw(helpers.get_dtypes("valid", full=False))
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
            "coo_indices": coo_ind,
            "values": np.array(val, dtype=val_dtype),
            "dense_shape": shp,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=False,
        all_as_kwargs_np_method={},
        fw=fw,
        class_name="SparseArray",
        method_name="to_dense_array",
    )
