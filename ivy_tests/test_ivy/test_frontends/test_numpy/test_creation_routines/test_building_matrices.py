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
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    dtype, x, k = dtype_and_x_k
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        v=x[0],
        k=k,
    )


@handle_frontend_test(
    fn_tree="numpy.vander",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.tuples(
            helpers.ints(min_value=1, max_value=10),
        ),
        large_abs_safety_factor=15,
        small_abs_safety_factor=15,
        safety_factor_scale="log",
    ),
    N=st.integers(min_value=1, max_value=10) | st.none(),
    increasing=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_vander(
    *, fn_tree, dtype_and_x, N, increasing, test_flags, backend_fw, frontend, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        x=x[0],
        N=N,
        increasing=increasing,
    )


@st.composite
def _diag_flat_helper(draw):
    x_shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=2, min_dim_size=1, max_dim_size=10
        )
    )
    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=x_shape,
            small_abs_safety_factor=2,
            large_abs_safety_factor=2,
            safety_factor_scale="log",
        )
    )
    k = draw(helpers.ints(min_value=-5, max_value=5))

    return dtype_and_x[0], dtype_and_x[1], k


# diagflat
@handle_frontend_test(
    fn_tree="numpy.diagflat",
    dtype_and_x_k=_diag_flat_helper(),
)
def test_numpy_diagflat(
    dtype_and_x_k,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x, k = dtype_and_x_k
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        v=x[0],
        k=k,
    )
