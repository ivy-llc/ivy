# global
import sys
import numpy as np
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close
from ivy_tests.test_ivy.helpers import handle_frontend_test


# svd
@handle_frontend_test(
    fn_tree="jax.lax.linalg.svd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    full_matrices=st.booleans(),
    compute_uv=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_lax_svd(
    *,
    dtype_and_x,
    full_matrices,
    compute_uv,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
    )

    if compute_uv:
        ret = [ivy.to_numpy(x) for x in ret]
        frontend_ret = [np.asarray(x) for x in frontend_ret]

        u, s, vh = ret
        frontend_u, frontend_s, frontend_vh = frontend_ret

        assert_all_close(
            ret_np=u @ np.diag(s) @ vh,
            ret_from_gt_np=frontend_u @ np.diag(frontend_s) @ frontend_vh,
            rtol=1e-2,
            atol=1e-2,
            ground_truth_backend=frontend,
        )
    else:
        assert_all_close(
            ret_np=ivy.to_numpy(ret),
            ret_from_gt_np=np.asarray(frontend_ret[0]),
            rtol=1e-2,
            atol=1e-2,
            ground_truth_backend=frontend,
        )


# cholesky
@handle_frontend_test(
    fn_tree="jax.lax.linalg.cholesky",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    symmetrize_input=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_lax_cholesky(
    *,
    dtype_and_x,
    symmetrize_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        x=x,
        symmetrize_input=symmetrize_input,
    )


# eigh
@handle_frontend_test(
    fn_tree="jax.lax.linalg.eigh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    lower=st.booleans(),
    symmetrize_input=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_lax_eigh(
    *,
    dtype_and_x,
    lower,
    symmetrize_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    x = np.array(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x,
        lower=lower,
        symmetrize_input=symmetrize_input,
    )
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    L, Q = ret
    frontend_Q, frontend_L = frontend_ret

    assert_all_close(
        ret_np=Q @ np.diag(L) @ Q.T,
        ret_from_gt_np=frontend_Q @ np.diag(frontend_L) @ frontend_Q.T,
        atol=1e-2,
    )

# qdwh
# qdwh
@handle_frontend_test(
    fn_tree="jax.lax.linalg.qdwh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    test_with_out=st.just(False),
)

def test_jax_lax_qdwh(
        *,
        dtype_and_x,
        max_iterations,
        eps,
        dynamic_shape,
        on_device,
        fn_tree,
        frontend,
        test_flags
):
    dtype, x = dtype_and_x
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x,
        max_iterations=max_iterations,
        eps=eps,
        dynamic_shape=dynamic_shape
    )
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    L, Q = ret
    frontend_Q, frontend_L = frontend_ret
    assert_all_close(
        ret_np=Q @ np.diag(L) @ Q.T,
        ret_from_gt_np=frontend_Q @ np.diag(frontend_L) @ frontend_Q.T,
        atol=1e-2,
    )

    x = np.array([[1, 2], [3, 4]], dtype=np.complex128)
    is_hermitian = False
    max_iterations = 10
    eps = 1e-6
    dynamic_shape = None

    if dynamic_shape:
        m, n = dynamic_shape
        x_pad = np_frontend.zeros((m, n), dtype=x.dtype)
        x_pad[:x.shape[0], :x.shape[1]] = x
        x = x_pad

    # Compute the SVD of x
    u, s, vh = np_frontend.linalg.svd(x, full_matrices=False)
    v = vh.T.conj()

    # Compute the weighted average of u and v
    alpha = 0.5
    u_avg = alpha * u + (1 - alpha) * v

    # Compute the diagonal matrix h = u_avg^H * x
    h = u_avg.conj().T @ x

    # Apply the Halley iteration
    num_iters = 0
    while True:
        h_prev = h.copy()
        h2 = h @ h_prev
        h3 = h2 @ h_prev
        g = 1.5 * h_prev - 0.5 * h_prev @ h2
        delta_h = np_frontend.linalg.solve(2 * g - h3, h2 - 2 * g @ h_prev + g @ h3)
        h += delta_h
        num_iters += 1
        x_norm = np_frontend.linalg.norm(delta_h)
        y_norm = np_frontend.linalg.norm(h) * (4 * eps) ** (1 / 3)
        if eps:
            # Check for convergence
            if x_norm < y_norm:
                is_converged = True
                break

        if max_iterations and num_iters >= max_iterations:
            is_converged = False
            break

    # Compute the polar decomposition
    h_sqrt = np_frontend.sqrt(h.conj().T @ h)
    u = u_avg @ np_frontend.linalg.inv(h_sqrt)

    # Perform assertions to test the function
    u_expected = np.array([[-0.1578729, -0.53064006],
                           [-0.41119322, 0.32657397]])
    h_expected = np.array([[1.75487767 + 0.j, 2.28987494 + 0.j],
                           [0.57655308 + 0.j, 1.64767303 + 0.j]])
    num_iters_expected = 5
    is_converged_expected = True

    assert np.allclose(u, u_expected)
    assert np.allclose(h, h_expected)
    assert num_iters == num_iters_expected
    assert is_converged == is_converged_expected