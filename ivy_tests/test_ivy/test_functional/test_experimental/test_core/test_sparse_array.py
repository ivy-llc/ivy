# global
from hypothesis import strategies as st

# local
import ivy
import numpy as np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method


# --- Helpers --- #
# --------------- #


@st.composite
def _sparse_bsc_indices_values_shape(draw):
    nblockrows = draw(helpers.ints(min_value=2, max_value=5))
    nblockcols = draw(helpers.ints(min_value=2, max_value=5))

    dim1 = draw(helpers.ints(min_value=2, max_value=5))
    dim2 = draw(helpers.ints(min_value=3, max_value=5))

    value_dtype = draw(helpers.get_dtypes("numeric", full=False))[0]

    ccol_indices, row_indices, values = (
        [0],
        [],
        [
            [
                [],
            ],
        ],
    )
    for _ in range(dim2):
        index = draw(
            helpers.ints(
                min_value=max(ccol_indices[-1] + 1, 1),
                max_value=ccol_indices[-1] + dim1,
            )
        )
        cur_num_elem = index - ccol_indices[-1]
        row_indices += list(range(cur_num_elem))
        ccol_indices.append(index)

    shape = (dim1 * nblockrows, dim2 * nblockcols)
    values = draw(
        helpers.array_values(
            dtype=value_dtype,
            shape=(ccol_indices[-1], nblockrows, nblockcols),
            min_value=0,
        )
    )

    return ccol_indices, row_indices, value_dtype, values, shape


@st.composite
def _sparse_bsr_indices_values_shape(draw):
    nblockrows = draw(helpers.ints(min_value=2, max_value=5))
    nblockcols = draw(helpers.ints(min_value=2, max_value=5))

    dim1 = draw(helpers.ints(min_value=3, max_value=5))
    dim2 = draw(helpers.ints(min_value=2, max_value=5))

    value_dtype = draw(helpers.get_dtypes("numeric", full=False))[0]

    crow_indices, col_indices, values = (
        [0],
        [],
        [
            [
                [],
            ],
        ],
    )
    for _ in range(dim1):
        index = draw(
            helpers.ints(
                min_value=max(crow_indices[-1] + 1, 1),
                max_value=crow_indices[-1] + dim2,
            )
        )
        cur_num_elem = index - crow_indices[-1]
        col_indices += list(range(cur_num_elem))
        crow_indices.append(index)

    shape = (dim1 * nblockrows, dim2 * nblockcols)
    values = draw(
        helpers.array_values(
            dtype=value_dtype,
            shape=(crow_indices[-1], nblockrows, nblockcols),
            min_value=0,
        )
    )

    return crow_indices, col_indices, value_dtype, values, shape


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


# --- Main --- #
# ------------ #


# adding sparse array to dense array
@handle_method(
    init_tree="ivy.array",
    method_tree="Array.__add__",
    sparse_data=_sparse_coo_indices_values_shape(),
)
def test_array_add_sparse(
    sparse_data,
    method_name,
    class_name,
    on_device,
):
    coo_ind, val_dtype, val, shp = sparse_data

    # set backed to 'torch' as this is the only backend which supports sparse arrays
    ivy.set_backend("torch")

    # initiate a sparse array
    sparse_inst = ivy.sparse_array.SparseArray(
        coo_indices=coo_ind,
        values=val,
        dense_shape=shp,
        format="coo",
    )

    # create an Array instance
    array_class = getattr(ivy, class_name)
    x = np.random.random_sample(shp)
    x = ivy.array(x, dtype=val_dtype, device=on_device)

    # call add method
    add_method = getattr(array_class, method_name)
    res = add_method(x, sparse_inst)

    # make sure the result is an Array instance
    assert isinstance(res, array_class)


# bsc - to_dense_array
@handle_method(
    method_tree="SparseArray.to_dense_array",
    sparse_data=_sparse_bsc_indices_values_shape(),
    method_num_positional_args=st.just(0),  # TODO should not be hardcoded
    init_num_positional_args=st.just(0),  # TODO should not be hardcoded
)
def test_sparse_bsc(
    sparse_data,
    class_name,
    method_name,
    on_device,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    ccol_indices, row_indices, value_dtype, values, shape = sparse_data
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        init_input_dtypes=["int64", "int64", value_dtype],
        init_all_as_kwargs_np={
            "ccol_indices": ccol_indices,
            "row_indices": row_indices,
            "values": values,
            "dense_shape": shape,
            "format": "bsc",
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


# bsr - to_dense_array
@handle_method(
    method_tree="SparseArray.to_dense_array",
    sparse_data=_sparse_bsr_indices_values_shape(),
    method_num_positional_args=st.just(0),  # TODO should not be hardcoded
    init_num_positional_args=st.just(0),  # TODO should not be hardcoded
)
def test_sparse_bsr(
    sparse_data,
    class_name,
    method_name,
    on_device,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    crow_indices, col_indices, value_dtype, values, shape = sparse_data
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        on_device=on_device,
        method_flags=method_flags,
        init_input_dtypes=["int64", "int64", value_dtype],
        init_all_as_kwargs_np={
            "crow_indices": crow_indices,
            "col_indices": col_indices,
            "values": values,
            "dense_shape": shape,
            "format": "bsr",
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


# coo - to_dense_array
@handle_method(
    method_tree="SparseArray.to_dense_array",
    sparse_data=_sparse_coo_indices_values_shape(),
    method_num_positional_args=st.just(0),  # TODO should not be hardcoded
    init_num_positional_args=st.just(0),  # TODO should not be hardcoded
)
def test_sparse_coo(
    sparse_data,
    class_name,
    method_name,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
    ground_truth_backend,
):
    coo_ind, val_dtype, val, shp = sparse_data
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        init_input_dtypes=["int64", val_dtype],
        init_all_as_kwargs_np={
            "coo_indices": coo_ind,
            "values": val,
            "dense_shape": shp,
            "format": "coo",
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


# csc - to_dense_array
@handle_method(
    method_tree="SparseArray.to_dense_array",
    sparse_data=_sparse_csc_indices_values_shape(),
    method_num_positional_args=st.just(0),  # TODO should not be hardcoded
    init_num_positional_args=st.just(0),  # TODO should not be hardcoded
)
def test_sparse_csc(
    sparse_data,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    on_device,
    method_flags,
):
    ccol_indices, row_indices, value_dtype, values, shape = sparse_data
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        init_input_dtypes=["int64", "int64", value_dtype],
        init_all_as_kwargs_np={
            "ccol_indices": ccol_indices,
            "row_indices": row_indices,
            "values": values,
            "dense_shape": shape,
            "format": "csc",
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


# csr - to_dense_array
@handle_method(
    method_tree="SparseArray.to_dense_array",
    sparse_data=_sparse_csr_indices_values_shape(),
    method_num_positional_args=st.just(0),  # TODO should not be hardcoded
    init_num_positional_args=st.just(0),  # TODO should not be hardcoded
)
def test_sparse_csr(
    sparse_data,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    on_device,
    method_flags,
):
    crow_indices, col_indices, value_dtype, values, shape = sparse_data
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        init_input_dtypes=["int64", "int64", value_dtype],
        init_all_as_kwargs_np={
            "crow_indices": crow_indices,
            "col_indices": col_indices,
            "values": values,
            "dense_shape": shape,
            "format": "csr",
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )
