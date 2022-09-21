# global
# import ivy
import sys
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# cholesky
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1])) != 0
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.cholesky"
    ),
)
def test_numpy_cholesky(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    fw,
):
    dtype, x = dtype_and_x
    x = np.asarray(x, dtype=dtype)
    x = (
        np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    )  # make symmetric positive-definite

    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.cholesky",
        rtol=1e-02,
        a=np.array(x, dtype=dtype),
    )


# qr
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        min_value=2,
        max_value=5,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.qr"
    ),
    mode=st.sampled_from(("reduced", "complete")),
)
def test_numpy_qr(
    dtype_and_x,
    mode,
    as_variable,
    native_array,
    num_positional_args,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.qr",
        rtol=1e-02,
        a=np.array(x, dtype=dtype),
        mode=mode,
    )


# svd
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.svd"
    ),
    fm=st.booleans(),
)
def test_numpy_svd(
    dtype_and_x,
    fm,
    as_variable,
    native_array,
    num_positional_args,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.svd",
        a=np.array(x, dtype=dtype),
        full_matrices=fm,
    )
