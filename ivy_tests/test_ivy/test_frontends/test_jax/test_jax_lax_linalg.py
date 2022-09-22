# global
import sys
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args, assert_all_close


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
        fn_name="ivy.functional.frontends.jax.lax.linalg.cholesky"
    ),
    symmetrize_input=st.booleans(),
)
def test_jax_lax_cholesky(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    fw,
    symmetrize_input,
):
    dtype, x = dtype_and_x
    x = np.array(x, dtype=dtype)
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.linalg.cholesky",
        rtol=1e-02,
        x=x,
        symmetrize_input=symmetrize_input,
    )


# eigh
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
        and x[0] != "float16"
        and x[0] != "bfloat16"
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.linalg.eigh"
    ),
    lower=st.booleans(),
    symmetrize_input=st.booleans(),
)
def test_jax_lax_eigh(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    fw,
    lower,
    symmetrize_input,
):
    dtype, x = dtype_and_x
    x = np.array(x, dtype=dtype)
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.linalg.eigh",
        test_values=False,
        x=x,
        lower=lower,
        symmetrize_input=symmetrize_input,
    )
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    Q, L = ret
    frontend_Q, frontend_L = frontend_ret

    assert_all_close(
        ret_np=Q @ np.diag(L) @ Q.T,
        ret_from_gt_np=frontend_Q @ np.diag(frontend_L) @ frontend_Q.T,
        atol=1e-05,
    )
