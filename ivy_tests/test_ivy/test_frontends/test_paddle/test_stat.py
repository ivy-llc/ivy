# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)


# mean
@handle_frontend_test(
    fn_tree="paddle.mean",
    dtype_and_x=_statistical_dtype_values(function="mean"),
    keepdim=st.booleans(),
    test_with_out=st.just(True),
)
def test_paddle_mean(
    *,
    dtype_and_x,
    keepdim,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x, axis = dtype_and_x[:3]
    test_flags.num_positional_args = len(dtype_and_x) - 2
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        input=x[0],
        axis=axis,
        keepdim=keepdim,
    )


# median
@handle_frontend_test(
    fn_tree="paddle.median",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_value=-1e10,
        max_value=1e10,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_paddle_median(
    dtype_x_and_axis, keepdim, backend_fw, frontend, test_flags, fn_tree
):
    input_dtypes, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        axis=axis,
        keepdim=keepdim,
    )


@handle_frontend_test(
    fn_tree="paddle.nanmedian",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_value=-1e10,
        max_value=1e10,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_paddle_nanmedian(
    dtype_x_and_axis,
    keepdim,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    input_dtypes, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        axis=axis,
        keepdim=keepdim,
    )


# numel
@handle_frontend_test(
    fn_tree="paddle.numel",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_numel(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# std
@handle_frontend_test(
    fn_tree="paddle.std",
    dtype_and_x=_statistical_dtype_values(function="std"),
    unbiased=st.booleans(),
    keepdim=st.booleans(),
)
def test_paddle_std(
    *,
    unbiased,
    dtype_and_x,
    keepdim,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x, axis, _ = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        axis=axis,
        unbiased=unbiased,
        keepdim=keepdim,
    )


# var
@handle_frontend_test(
    fn_tree="paddle.var",
    dtype_and_x=_statistical_dtype_values(function="var"),
    unbiased=st.booleans(),
    keepdim=st.booleans(),
)
def test_paddle_var(
    *,
    unbiased,
    dtype_and_x,
    keepdim,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x, axis, _ = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        axis=axis,
        unbiased=unbiased,
        keepdim=keepdim,
    )
