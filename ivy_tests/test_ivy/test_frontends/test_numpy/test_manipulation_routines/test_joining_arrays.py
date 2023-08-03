# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


@st.composite
def _arrays_idx_n_dtypes(draw):
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    input_dtypes, x, casting, dtype = draw(
        np_frontend_helpers.dtypes_values_casting_dtype(
            arr_func=[
                lambda: helpers.dtype_and_values(
                    available_dtypes=helpers.get_dtypes("numeric"),
                    shape=shape,
                    num_arrays=num_arrays,
                    shared_dtype=True,
                )
            ],
        ),
    )
    axis = draw(helpers.get_axis(shape=shape, force_int=True))
    return x, input_dtypes, axis, casting, dtype


# concat
@handle_frontend_test(
    fn_tree="numpy.concatenate",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
)
def test_numpy_concatenate(
    xs_n_input_dtypes_n_unique_idx,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    xs, input_dtypes, unique_idx, casting, dtype = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-01,
        arrays=xs,
        axis=unique_idx,
        casting=casting,
        dtype=dtype,
        out=None,
    )


# stack
@handle_frontend_test(
    fn_tree="numpy.stack",
    dtype_and_x=_arrays_idx_n_dtypes(),
)
def test_numpy_stack(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    xs, input_dtypes, unique_idx, _, _ = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        arrays=xs,
        axis=unique_idx,
    )


# vstack
@handle_frontend_test(
    fn_tree="numpy.vstack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shared_dtype=True,
        num_arrays=helpers.ints(min_value=2, max_value=10),
        shape=helpers.get_shape(
            min_num_dims=1,
        ),
    ),
)
def test_numpy_vstack(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tup=xs,
    )


# hstack
@handle_frontend_test(
    fn_tree="numpy.hstack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shared_dtype=True,
        num_arrays=helpers.ints(min_value=2, max_value=10),
        shape=helpers.get_shape(
            min_num_dims=1,
        ),
    ),
)
def test_numpy_hstack(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tup=xs,
    )


#block
@handle_frontend_test(
    fn_tree="numpy.block",
    dtype_and_x=helpers.get_dtypes("lists")
)
def test_numpy_block(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    xs, input_dtypes = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        arrays=xs,

    )
