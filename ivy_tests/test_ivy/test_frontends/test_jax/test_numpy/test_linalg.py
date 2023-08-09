# global
import sys
import numpy as np

from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import (
    assert_all_close,
    handle_frontend_test,
    update_backend,
)
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
    _matrix_rank_helper,
)
from ivy_tests.test_ivy.helpers.hypothesis_helpers.general_helpers import (
    matrix_is_stable,
)


# svd
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.svd",
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
def test_jax_svd(
    *,
    dtype_and_x,
    full_matrices,
    compute_uv,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
    )

    if compute_uv:
        with update_backend(backend_fw) as ivy_backend:
            ret = [ivy_backend.to_numpy(x) for x in ret]
        frontend_ret = [np.asarray(x) for x in frontend_ret]

        u, s, vh = ret
        frontend_u, frontend_s, frontend_vh = frontend_ret

        assert_all_close(
            ret_np=u @ np.diag(s) @ vh,
            ret_from_gt_np=frontend_u @ np.diag(frontend_s) @ frontend_vh,
            rtol=1e-2,
            atol=1e-2,
            backend=backend_fw,
            ground_truth_backend=frontend,
        )
    else:
        with update_backend(backend_fw) as ivy_backend:
            ret = ivy_backend.to_numpy(ret)
        assert_all_close(
            ret_np=ret,
            ret_from_gt_np=np.asarray(frontend_ret[0]),
            rtol=1e-2,
            atol=1e-2,
            backend=backend_fw,
            ground_truth_backend=frontend,
        )


# det
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.det",
    dtype_and_x=_get_dtype_and_matrix(),
    test_with_out=st.just(False),
)
def test_jax_det(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-04,
        atol=1e-04,
        a=x[0],
    )


# eig
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.eig",
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
def test_jax_eig(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, x = dtype_and_x
    x = np.array(x[0], dtype=dtype[0])
    """make symmetric positive-definite since ivy does not support complex data dtypes
    currently."""
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        a=x,
    )

    with update_backend(backend_fw) as ivy_backend:
        ret = [ivy_backend.to_numpy(x).astype(np.float64) for x in ret]
    frontend_ret = [x.astype(np.float64) for x in frontend_ret]

    L, Q = ret
    frontend_L, frontend_Q = frontend_ret

    assert_all_close(
        ret_np=Q @ np.diag(L) @ Q.T,
        ret_from_gt_np=frontend_Q @ np.diag(frontend_L) @ frontend_Q.T,
        atol=1e-02,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )


# eigh
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.eigh",
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
    UPLO=st.sampled_from(("L", "U")),
    symmetrize_input=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_eigh(
    *,
    dtype_and_x,
    UPLO,
    symmetrize_input,
    backend_fw,
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
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        a=x,
        UPLO=UPLO,
        symmetrize_input=symmetrize_input,
    )
    with update_backend(backend_fw) as ivy_backend:
        ret = [ivy_backend.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    L, Q = ret
    frontend_L, frontend_Q = frontend_ret

    assert_all_close(
        ret_np=Q @ np.diag(L) @ Q.T,
        ret_from_gt_np=frontend_Q @ np.diag(frontend_L) @ frontend_Q.T,
        atol=1e-02,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )


# inv
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.inv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        shape=helpers.ints(min_value=1, max_value=10).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    test_with_out=st.just(False),
)
def test_jax_inv(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        rtol=1e-01,
        atol=1e-01,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# eigvalsh
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.eigvalsh",
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
    UPLO=st.sampled_from(("L", "U")),
    test_with_out=st.just(False),
)
def test_jax_eigvalsh(
    *,
    dtype_and_x,
    UPLO,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-02,
        a=x,
        UPLO=UPLO,
    )


# qr
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.qr",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        min_value=2,
        max_value=5,
    ),
    mode=st.sampled_from(("reduced", "complete")),
    test_with_out=st.just(False),
)
def test_jax_qr(
    *,
    dtype_and_x,
    mode,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        test_values=False,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=np.asarray(x[0], dtype[0]),
        mode=mode,
    )
    with update_backend(backend_fw) as ivy_backend:
        ret = [ivy_backend.to_numpy(x).astype(np.float64) for x in ret]
    frontend_ret = [x.astype(np.float64) for x in frontend_ret]

    Q, R = ret
    frontend_Q, frontend_R = frontend_ret

    assert_all_close(
        ret_np=Q @ R,
        ret_from_gt_np=frontend_Q @ frontend_R,
        atol=1e-02,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )


# eigvals
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.eigvals",
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
def test_jax_eigvals(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    x = np.array(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        a=x,
    )


# cholesky
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.cholesky",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(x[1][0]) != 0
    ),
    test_with_out=st.just(False),
)
def test_jax_cholesky(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        a=x,
    )


# slogdet
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.slogdet",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_value=100,
        min_value=-100,
        shape=st.tuples(
            st.shared(st.integers(1, 5), key="sq"),
            st.shared(st.integers(1, 5), key="sq"),
        ),
        num_arrays=1,
    ),
    test_with_out=st.just(False),
)
def test_jax_slogdet(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-4,
        rtol=1e-4,
        a=x[0],
    )


# matrix_rank
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.matrix_rank",
    dtype_x_hermitian_atol_rtol=_matrix_rank_helper(),
    test_with_out=st.just(False),
)
def test_jax_matrix_rank(
    *,
    dtype_x_hermitian_atol_rtol,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, hermitian, atol, rtol = dtype_x_hermitian_atol_rtol
    assume(matrix_is_stable(x, cond_limit=10))
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        M=x,
        tol=atol,
    )


# solve
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.solve",
    x=helpers.get_first_solve_matrix(adjoint=False),
    y=helpers.get_second_solve_matrix(),
    test_with_out=st.just(False),
)
def test_jax_solve(
    *,
    x,
    y,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype1, x1, _ = x
    input_dtype2, x2 = y
    helpers.test_frontend_function(
        input_dtypes=[input_dtype1, input_dtype2],
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        a=x1,
        b=x2,
    )


@st.composite
def norm_helper(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            safety_factor_scale="log",
            large_abs_safety_factor=2,
        )
    )
    axis = draw(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        )
    )
    if type(axis) in [tuple, list]:
        if len(axis) == 2:
            ord_param = draw(
                st.sampled_from(["fro", "nuc", 1, 2, -1, -2, np.inf, -np.inf])
            )
        else:
            axis = axis[0]
            ord_param = draw(st.sampled_from([0, 1, 2, -1, -2, np.inf, -np.inf]))
    else:
        ord_param = draw(st.sampled_from([0, 1, 2, -1, -2, np.inf, -np.inf]))
    keepdims = draw(st.booleans())
    return dtype, x, ord_param, axis, keepdims


# norm
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.norm",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=5,
        min_axis=-2,
        max_axis=1,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    keepdims=st.booleans(),
    ord=st.sampled_from([None, np.inf, -np.inf, 1, -1, 2, -2]),
    test_with_out=st.just(False),
)
def test_jax_norm(
    dtype_values_axis,
    keepdims,
    ord,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x, axis = dtype_values_axis

    if len(np.shape(x)) == 1:
        axis = None

    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        ord=ord,
        axis=axis,
        keepdims=keepdims,
        atol=1e-1,
        rtol=1e-1,
    )


# matrix_power
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.matrix_power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        shape=helpers.ints(min_value=1, max_value=10).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    n=helpers.ints(min_value=1, max_value=8),
    test_with_out=st.just(False),
)
def test_jax_matrix_power(
    *,
    dtype_and_x,
    n,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        rtol=1e-01,
        atol=1e-01,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a=np.asarray(x[0], dtype=dtype[0]),
        n=n,
    )


# tensorsolve
@st.composite
def _get_solve_matrices(draw):
    # batch_shape, random_size, shared

    # float16 causes a crash when filtering out matrices
    # for which `np.linalg.cond` is large.
    input_dtype_strategy = st.shared(
        st.sampled_from(draw(helpers.get_dtypes("float"))).filter(
            lambda x: "float16" not in x
        ),
        key="shared_dtype",
    )
    input_dtype = draw(input_dtype_strategy)

    dim = draw(helpers.ints(min_value=2, max_value=5))

    first_matrix = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=(dim, dim, dim, dim),
            min_value=1.2,
            max_value=5,
        ).filter(lambda x: np.linalg.det(x.reshape((dim**2, dim**2))) != 0)
    )

    second_matrix = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=(dim, dim),
            min_value=1.2,
            max_value=3,
        )
    )

    return input_dtype, first_matrix, second_matrix


@handle_frontend_test(
    fn_tree="jax.numpy.linalg.tensorsolve",
    a_and_b=_get_solve_matrices(),
    test_with_out=st.just(False),
)
def test_jax_tensorsolve(
    *,
    a_and_b,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, y = a_and_b
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x,
        b=y,
        atol=1e-2,
        rtol=1e-2,
    )


# pinv
@handle_frontend_test(
    fn_tree="jax.numpy.linalg.pinv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
    rcond=st.floats(1e-5, 1e-3),
)
def test_jax_pinv(
    dtype_and_x,
    frontend,
    fn_tree,
    test_flags,
    backend_fw,
    rcond,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        a=x[0],
        rcond=rcond,
        atol=1e-1,
        rtol=1e-1,
    )


# tensorinv
@st.composite
def _get_inv_square_matrices(draw):
    dim_size = draw(helpers.ints(min_value=1, max_value=9))

    batch_shape = draw(st.sampled_from([2, 4, 6, 8]))

    generated_shape = (dim_size,) * batch_shape
    generated_ind = int(np.floor(len(generated_shape) / 2))

    handpicked_shape, handpicked_ind = draw(
        st.sampled_from([[(24, 6, 4), 1], [(8, 3, 6, 4), 2], [(6, 7, 8, 16, 21), 3]])
    )

    shape, ind = draw(
        st.sampled_from(
            [(generated_shape, generated_ind), (handpicked_shape, handpicked_ind)]
        )
    )

    input_dtype = draw(
        helpers.get_dtypes("float", index=1, full=False).filter(
            lambda x: x not in ["float16", "bfloat16"]
        )
    )
    invertible = False
    while not invertible:
        a = draw(
            helpers.array_values(
                dtype=input_dtype[0],
                shape=shape,
                large_abs_safety_factor=24,
                small_abs_safety_factor=24,
                safety_factor_scale="log",
            ).filter(lambda x: helpers.matrix_is_stable(x))
        )
        try:
            np.linalg.tensorinv(a, ind)
            invertible = True
        except np.linalg.LinAlgError:
            pass

    return input_dtype, a, ind


@handle_frontend_test(
    fn_tree="jax.numpy.linalg.tensorinv", params=_get_inv_square_matrices()
)
def test_jax_tensorinv(
    *,
    params,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, ind = params
    helpers.test_frontend_function(
        input_dtypes=dtype,
        rtol=1e-01,
        atol=1e-01,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x,
        ind=ind,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.linalg.cond",
    dtype_x_p=helpers.cond_data_gen_helper(),
    test_with_out=st.just(False),
)
def test_jax_cond(
    *,
    dtype_x_p,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    dtype, x = dtype_x_p
    helpers.test_frontend_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        rtol=1e-01,
        atol=1e-01,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        p=x[1],
    )


# multi_dot
@handle_frontend_test(
    fn_tree="jax.lax.linalg.multi_dot",
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
def test_jax_lax_multi_dot(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        args=(x,),
    )

    ret = ivy.to_numpy(ret)
    frontend_ret = np.asarray(frontend_ret)

    assert_all_close(
        ret_np=ret,
        ret_from_gt_np=frontend_ret,
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
    )
