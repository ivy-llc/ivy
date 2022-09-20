# global
import sys
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import _get_dtype_and_matrix


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
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.linalg.eigh"
    ),
    lower=st.booleans(),
    symmetrize_input=st.booleans(),
    sort_eigenvalues=st.booleans(),
)
def test_jax_lax_eigh(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    lower,
    symmetrize_input,
    sort_eigenvalues,
):
    dtype, x = dtype_and_x
    x = np.array(x, dtype=dtype)
  
    results = helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.linalg.eigh",
        x=x,
        test_values=False,
        return_flat_np_arrays=True,
        lower=lower
        symmetrize_input=symmetrize_input,
        sort_eigenvalues=sort_eigenvalues
    )
    if results is None:
        return
    ret_np_flat, ret_from_np_flat = results
    eigenvalues_np, eigenvectors_np = ret_np_flat
    reconstructed_np = None
    for eigenvalue, eigenvector in zip(eigenvalues_np, eigenvectors_np):
        if reconstructed_np is not None:
            reconstructed_np += eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
        else:
            reconstructed_np = eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
    eigenvalues_from_np, eigenvectors_from_np = ret_from_np_flat
    reconstructed_from_np = None
    for eigenvalue, eigenvector in zip(eigenvalues_from_np, eigenvectors_from_np):
        if reconstructed_from_np is not None:
            reconstructed_from_np += eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
        else:
            reconstructed_from_np = eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
    # value test
    helpers.assert_all_close(
        reconstructed_np, reconstructed_from_np, rtol=1e-1, atol=1e-2
    )
