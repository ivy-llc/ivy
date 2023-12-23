# global
import numpy as np
from hypothesis import strategies as st
from numpy import triu, tril

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


# unravel_index
@st.composite
def max_value_as_shape_prod(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
        )
    )
    dtype_and_x = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("valid"),
            min_value=0,
            max_value=np.prod(shape) - 1,
        )
    )
    return dtype_and_x, shape


@handle_frontend_test(
    fn_tree="numpy.diag_indices",
    n=helpers.ints(min_value=1, max_value=10),
    ndim=helpers.ints(min_value=2, max_value=10),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_diag_indices(
    n,
    ndim,
    dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n,
        ndim=ndim,
    )


@handle_frontend_test(
    fn_tree="numpy.indices",
    dimensions=helpers.get_shape(min_num_dims=1),
    dtype=helpers.get_dtypes(kind="float", full=False),
    sparse=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_indices(
    *,
    dimensions,
    dtype,
    sparse,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        dimensions=dimensions,
        dtype=dtype[0],
        sparse=sparse,
    )


@handle_frontend_test(
    fn_tree="numpy.mask_indices",
    n=helpers.ints(min_value=3, max_value=10),
    mask_func=st.sampled_from([triu, tril]),
    k=helpers.ints(min_value=-5, max_value=5),
    input_dtype=helpers.get_dtypes("numeric"),
    test_with_out=st.just(False),
    number_positional_args=st.just(2),
)
def test_numpy_mask_indices(
    n,
    mask_func,
    k,
    input_dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n,
        mask_func=mask_func,
        k=k,
    )


@handle_frontend_test(
    fn_tree="numpy.tril_indices",
    n=helpers.ints(min_value=1, max_value=10),
    m=helpers.ints(min_value=1, max_value=10),
    k=st.integers(min_value=-10, max_value=10),
    test_with_out=st.just(False),
)
def test_numpy_tril_indices(
    *,
    n,
    m,
    k,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=["int32"],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n,
        k=k,
        m=m,
    )


@handle_frontend_test(
    fn_tree="numpy.tril_indices_from",
    dtype_and_values=helpers.dtype_and_values(
        dtype=["float32"],
        min_dim_size=3,
        max_dim_size=3,
        min_num_dims=2,
        max_num_dims=2,
        array_api_dtypes=True,
    ),
    k=st.integers(min_value=-10, max_value=10),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_tril_indices_from(
    *,
    dtype_and_values,
    k,
    dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=values[0],
        k=k,
    )


@handle_frontend_test(
    fn_tree="numpy.unravel_index",
    dtype_x_shape=max_value_as_shape_prod(),
    test_with_out=st.just(False),
)
def test_numpy_unravel_index(
    *,
    dtype_x_shape,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype_and_x, shape = dtype_x_shape
    input_dtype, x = dtype_and_x[0], dtype_and_x[1]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        indices=x[0],
        shape=shape,
    )
