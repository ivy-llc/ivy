# global
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


# triangular_solve
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
        num_arrays=1,
    ),
    as_variable=helpers.array_bools(num_arrays=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.linalg.triangular_solve"
    ),
)
def test_jax_lax_linalg_triangular_solve(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    fw,
):
    dtype, x = dtype_and_x
    num_positional_args = 2
    mask = np.asarray(x, dtype=dtype) * 0.0
    iu = np.tril_indices(mask.shape[0])
    mask[iu] = 1.0
    a = x * mask
    b = x[0]

    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.linalg.triangular_solve",
        atol=0.1,
        a=np.array(a, dtype=dtype),
        b=np.array(b, dtype=dtype),
        left_side=False,
        lower=False,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=False
    )