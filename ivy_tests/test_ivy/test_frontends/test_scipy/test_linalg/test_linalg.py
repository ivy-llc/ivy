# TODO: uncomment after frontend is not required
# global
import ivy
import sys
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Helpers #
# ------- #


@st.composite
def _generate_eigh_tridiagonal_args(draw):
    dtype, alpha = draw(
        helpers.dtype_and_values(
            min_dim_size=2,
            min_num_dims=1,
            max_num_dims=1,
            min_value=2.0,
            max_value=5,
            available_dtypes=helpers.get_dtypes("float"),
        )
    )
    beta_shape = len(alpha[0]) - 1
    dtype, beta = draw(
        helpers.dtype_and_values(
            available_dtypes=dtype,
            shape=(beta_shape,),
            min_value=2.0,
            max_value=5,
        )
    )

    select = draw(st.sampled_from(("a", "i", "v")))
    if select == "a":
        select_range = None
    elif select == "i":
        range_slice = draw(
            st.slices(beta_shape).filter(
                lambda x: x.start
                and x.stop
                and x.step
                and x.start >= 0
                and x.stop >= 0
                and x.step >= 0
                and x.start < x.stop
            )
        )

        select_range = [range_slice.start, range_slice.stop]
    else:
        select_range = [-100, 100]

    eigvals_only = draw(st.booleans())
    tol = draw(st.floats(1e-5, 1e-3) | st.just(None))
    return dtype, alpha, beta, eigvals_only, select, select_range, tol


@st.composite
def _norm_helper(draw):
    def _matrix_norm_example():
        x_dtype, x = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                shape=helpers.get_shape(min_num_dims=2, max_num_dims=2),
                min_num_dims=2,
                max_num_dims=2,
                min_dim_size=1,
                max_dim_size=10,
                min_value=-1e4,
                max_value=1e4,
                large_abs_safety_factor=10,
                small_abs_safety_factor=10,
                safety_factor_scale="log",
            ),
        )
        ord = draw(st.sampled_from(["fro", "nuc"]))
        axis = (-2, -1)
        check_stable = True
        return x_dtype, x, axis, ord, check_stable

    def _vector_norm_example():
        x_dtype, x, axis = draw(
            helpers.dtype_values_axis(
                available_dtypes=helpers.get_dtypes("float"),
                min_num_dims=2,
                max_num_dims=5,
                min_dim_size=2,
                max_dim_size=10,
                valid_axis=True,
                force_int_axis=True,
                min_value=-1e04,
                max_value=1e04,
                large_abs_safety_factor=10,
                small_abs_safety_factor=10,
                safety_factor_scale="log",
            )
        )
        ints = draw(helpers.ints(min_value=1, max_value=2))
        floats = draw(helpers.floats(min_value=1, max_value=2))
        ord = draw(st.sampled_from([ints, floats, float("inf"), float("-inf")]))
        check_stable = False
        return x_dtype, x, axis, ord, check_stable

    is_vec_norm = draw(st.booleans())
    if is_vec_norm:
        return _vector_norm_example()
    return _matrix_norm_example()


# Tests #
# ----- #


# tril
@handle_frontend_test(
    fn_tree="scipy.linalg.tril",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    test_with_out=st.just(False),
)
def test_scipy_tril(
    dtype_and_x,
    k,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=x[0],
        k=k,
    )


# triu
@handle_frontend_test(
    fn_tree="scipy.linalg.triu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    test_with_out=st.just(False),
)
def test_scipy_triu(
    dtype_and_x,
    k,
    test_flags,
    frontend,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=x[0],
        k=k,
    )


# inv
@handle_frontend_test(
    fn_tree="scipy.linalg.inv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_abs_safety_factor=2,
        safety_factor_scale="log",
        shape=helpers.ints(min_value=2, max_value=20).map(lambda x: tuple([x, x])),
    ).filter(lambda x: np.linalg.cond(x[1][0].tolist()) < 1 / sys.float_info.epsilon),
    test_with_out=st.just(False),
)
def test_scipy_inv(
    dtype_and_x,
    test_flags,
    frontend,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# pinv
@handle_frontend_test(
    fn_tree="scipy.linalg.pinv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=2,
    ),
    test_with_out=st.just(False),
)
def test_scipy_pinv(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# kron
@handle_frontend_test(
    fn_tree="scipy.linalg.kron",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=10,
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_scipy_kron(dtype_and_x, frontend, test_flags, fn_tree, on_device, backend_fw):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        b=x[1],
    )


# eigh_tridiagonal
@handle_frontend_test(
    fn_tree="scipy.linalg.eigh_tridiagonal",
    all_args=_generate_eigh_tridiagonal_args(),
    test_with_out=st.just(False),
)
def test_scipy_eigh_tridiagonal(
    all_args,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, alpha, beta, eigvals_only, select, select_range, tol = all_args
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        alpha=alpha[0],
        beta=beta[0],
        eigvals_only=eigvals_only,
        select=select,
        select_range=select_range,
        tol=tol,
    )


# norm
@handle_frontend_test(
    fn_tree="scipy.linalg.norm",
    dtype_values=_norm_helper(),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
)
def test_scipy_norm(
    dtype_values,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x, axis, ord, _ = dtype_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        ord=ord,
        axis=axis,
        keepdims=keepdims,
    )


# svd
@handle_frontend_test(
    fn_tree="scipy.linalg.svd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0.1,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ),
    full_matrices=st.booleans(),
    compute_uv=st.booleans(),
    test_with_out=st.just(False),
)
def test_scipy_svd(
    dtype_and_x,
    full_matrices,
    compute_uv,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        test_values=False,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        full_matrices=full_matrices,
        compute_uv=compute_uv,
    )
    for u, v in zip(ret, ret_gt):
        u = ivy.to_numpy(ivy.abs(u))
        v = ivy.to_numpy(ivy.abs(v))
        helpers.value_test(ret_np_flat=u, ret_np_from_gt_flat=v, rtol=1e-04, atol=1e-04)


# svdvals
@handle_frontend_test(
    fn_tree="scipy.linalg.svdvals",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        min_num_dims=2,
    ),
    test_with_out=st.just(False),
)
def test_scipy_svdvals(
    dtype_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# lu_factor
@handle_frontend_test(
    fn_tree="scipy.linalg.lu_factor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        min_num_dims=2,
    ),
    overwrite_a=st.booleans(),
    check_finite=st.booleans(),
    test_with_out=st.just(False),
)
def test_scipy_lu_factor(
    dtype_and_x,
    overwrite_a,
    check_finite,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        test_values=False,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        overwrite_a=overwrite_a,
        check_finite=check_finite,
    )
