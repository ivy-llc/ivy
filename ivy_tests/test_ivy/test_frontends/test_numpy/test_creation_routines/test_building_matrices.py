# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import _diag_helper


# tril
@handle_frontend_test(
    fn_tree="numpy.tril",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    test_with_out=st.just(False),
)
def test_numpy_tril(
    dtype_and_x,
    k,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=x[0],
        k=k,
    )


# triu
@handle_frontend_test(
    fn_tree="numpy.triu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    test_with_out=st.just(False),
)
def test_numpy_triu(
    dtype_and_x,
    k,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=x[0],
        k=k,
    )


# tri
@handle_frontend_test(
    fn_tree="numpy.tri",
    rows=helpers.ints(min_value=3, max_value=10),
    cols=helpers.ints(min_value=3, max_value=10),
    k=helpers.ints(min_value=-10, max_value=10),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_tri(
    rows,
    cols,
    k,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        N=rows,
        M=cols,
        k=k,
        dtype=dtype[0],
    )


# diag
@handle_frontend_test(
    fn_tree="numpy.diag",
    dtype_and_x_k=_diag_helper(),
)
def test_numpy_diag(
    dtype_and_x_k,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, k = dtype_and_x_k
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        v=x[0],
        k=k,
    )
