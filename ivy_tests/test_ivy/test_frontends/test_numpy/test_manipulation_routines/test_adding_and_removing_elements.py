# global
from hypothesis import assume

# local
import ivy
from hypothesis import strategies as st
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# unique
@handle_frontend_test(
    fn_tree="numpy.unique",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        force_int_axis=True,
        valid_axis=True,
    ),
    return_index=st.booleans(),
    return_inverse=st.booleans(),
    return_counts=st.booleans(),
    none_axis=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_unique(
    *,
    dtype_x_axis,
    return_index,
    return_inverse,
    return_counts,
    none_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtypes, xs, axis = dtype_x_axis
    if none_axis:
        axis = None
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=xs[0],
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    )


# append
@handle_frontend_test(
    fn_tree="numpy.append",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shape=helpers.get_shape(
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
        ),
        shared_dtype=True,
        valid_axis=True,
        allow_neg_axes=True,
        force_int_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_numpy_append(
    dtype_values_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, values, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=values[0],
        values=values[1],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="numpy.trim_zeros",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=1, max_num_dims=1
    ),
    trim=st.sampled_from(["f", "b", "fb"]),
)
def test_numpy_trim_zeros(
    frontend,
    on_device,
    *,
    dtype_and_x,
    trim,
    fn_tree,
    test_flags,
    backend_fw,
):
    input_dtypes, x = dtype_and_x
    if ivy.current_backend_str() == "paddle":
        assume(input_dtypes[0] not in ["float16"])
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        filt=x[0],
        trim=trim,
    )
