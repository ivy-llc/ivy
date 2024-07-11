# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.array",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_with_out=st.just(False),
    test_with_copy=st.just(True),
)
def test_numpy_array(
    dtype_and_a,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        object=a,
        dtype=dtype[0],
    )


# asarray
@handle_frontend_test(
    fn_tree="numpy.asarray",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_with_out=st.just(False),
)
def test_numpy_asarray(
    dtype_and_a,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a,
        dtype=dtype[0],
    )


# copy
@handle_frontend_test(
    fn_tree="numpy.copy",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_with_out=st.just(False),
    test_with_copy=st.just(True),
)
def test_numpy_copy(
    dtype_and_a,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
    )


# frombuffer
@handle_frontend_test(
    fn_tree="numpy.frombuffer",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_with_out=st.just(False),
)
def test_numpy_frombuffer(
    dtype_and_a,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        buffer=a,
        dtype=dtype[0],
    )
