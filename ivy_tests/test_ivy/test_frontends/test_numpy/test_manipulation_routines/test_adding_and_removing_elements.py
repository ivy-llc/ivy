# global

# local
from hypothesis import strategies as st
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# shape
@handle_frontend_test(
    fn_tree="numpy.unique",
    xs_n_input_dtypes_n_unique_idx=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float")
    ),
)
def test_numpy_unique(
        *,
        xs_n_input_dtypes_n_unique_idx,
        on_device,
        fn_tree,
        frontend,
        test_flags,
):
    input_dtypes, xs = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=xs[0],
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )


@handle_frontend_test(
    fn_tree="numpy.trim_zeros",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=1
    ),
    trim=st.sampled_from(['f', 'b', 'fb'])
)
def test_numpy_trim_zeros(
        frontend,
        on_device,
        *,
        dtype_and_x,
        trim,
        fn_tree,
        test_flags,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        filt=x[0],
        trim=trim
    )
