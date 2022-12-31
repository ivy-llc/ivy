# global
import numpy as np
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
    matrix_is_stable,
)


# norm
@handle_frontend_test(
    fn_tree="numpy.linalg.norm",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=5,
        min_axis=-2,
        max_axis=1,
    ),
    keepdims=st.booleans(),
    ord=st.sampled_from([None, "fro", "nuc", "inf", "-inf", 0, 1, -1, 2, -2]),
)
def test_numpy_norm(
    dtype_values_axis,
    keepdims,
    ord,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_values_axis
    if len(np.shape(x)) == 1:
        axis = None
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        ord=ord,
        axis=axis,
        keepdims=keepdims,
    )


# matrix_rank
@handle_frontend_test(
    fn_tree="numpy.linalg.matrix_rank",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        min_value=-1e05,
        max_value=1e05,
    ),
    rtol=st.floats(allow_nan=False, allow_infinity=False) | st.just(None),
)
def test_numpy_matrix_rank(
    dtype_and_x,
    rtol,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        A=x[0],
        tol=rtol,
    )


# det
@handle_frontend_test(
    fn_tree="numpy.linalg.det",
    dtype_and_x=_get_dtype_and_matrix(),
)
def test_numpy_det(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        a=x[0],
    )


# slogdet
@handle_frontend_test(
    fn_tree="numpy.linalg.slogdet",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_value=5,
        min_value=2,
        shape=st.tuples(
            st.shared(st.integers(1, 5), key="sq"),
            st.shared(st.integers(1, 5), key="sq"),
        ),
        num_arrays=1,
        safety_factor_scale="log",
    ),
)
def test_numpy_slogdet(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    assume(matrix_is_stable(x[0]))
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        test_values=False,
    )
    for ret_f, ret_gtt in zip(ret, ret_gt):
        frontend_ret = ret_f
        frontend_ret_gt = ret_gtt
        ret_flattened = helpers.flatten_and_to_np(ret=frontend_ret)
        ret_gt_flattened = helpers.flatten_and_to_np(ret=frontend_ret_gt)
        helpers.value_test(
            ret_np_flat=ret_flattened,
            ret_np_from_gt_flat=ret_gt_flattened,
            rtol=1e-1,
            atol=1e-1,
            ground_truth_backend="numpy",
        )


@handle_frontend_test(
    fn_tree="numpy.trace",
    dtype_and_x=_get_dtype_and_matrix(),
)
def test_numpy_trace(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )
