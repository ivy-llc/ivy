# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.argsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_numpy_argsort(
    *,
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="numpy.sort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_numpy_sort(
    *,
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="numpy.msort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        min_dim_size=1,
        min_axis=-1,
        max_axis=0,
    ),
)
def test_numpy_msort(
    *,
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


@handle_frontend_test(
    fn_tree="numpy.sort_complex",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_dim_size=1,
        min_axis=-1,
        max_axis=0,
    ),
    test_with_out=st.just(False),
)
def test_numpy_sort_complex(
    *,
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        test_values=False,
    )


@handle_frontend_test(
    fn_tree="numpy.lexsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_numpy_lexsort(
    *,
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        keys=x[0],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="numpy.partition",
    dtype_x_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int64"],
        min_dim_size=1,
        max_num_dims=1,
        indices_same_dims=False,
        disable_random_axis=False,
        axis_zero=False,
        valid_bounds=True,
    ),
    test_with_out=st.just(False),
)
def test_numpy_partition(
    *,
    dtype_x_axis,
    frontend,
    test_flags,
    backend_fw,
    fn_tree,
    on_device,
):
    dtypes, x, kth, axis, _ = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        a=x,
        kth=kth,
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="numpy.argpartition",
    dtype_x_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int64"],
        min_dim_size=1,
        max_num_dims=1,
        indices_same_dims=False,
        disable_random_axis=False,
        axis_zero=False,
        valid_bounds=True,
    ),
    test_with_out=st.just(False),
)
def test_argpartition(
    *,
    dtype_x_axis,
    frontend,
    test_flags,
    backend_fw,
    fn_tree,
    on_device,
):
    dtypes, x, kth, axis, _ = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        a=x,
        kth=kth,
        axis=axis,
    )
