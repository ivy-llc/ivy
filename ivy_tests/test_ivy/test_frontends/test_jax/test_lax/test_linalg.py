# global
import sys
import numpy as np
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close
from ivy_tests.test_ivy.helpers import handle_frontend_test, BackendHandler


# cholesky
@handle_frontend_test(
    fn_tree="jax.lax.linalg.cholesky",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: (x, x)),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    symmetrize_input=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_cholesky(
    *,
    dtype_and_x,
    symmetrize_input,
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
    fw_ret, gt_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        x=x,
        symmetrize_input=symmetrize_input,
        test_values=False,
    )
    # ToDo: turn value test on when jax cholesky is fixed in issue
    # https: // github.com / google / jax / issues / 16185
    helpers.assertions.assert_same_type_and_shape([fw_ret, gt_ret])


# eigh
@handle_frontend_test(
    fn_tree="jax.lax.linalg.eigh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: (x, x)),
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
def test_jax_eigh(
    *,
    dtype_and_x,
    lower,
    symmetrize_input,
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
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x,
        lower=lower,
        symmetrize_input=symmetrize_input,
    )
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        ret = [ivy_backend.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    L, Q = ret
    frontend_Q, frontend_L = frontend_ret

    assert_all_close(
        ret_np=Q @ np.diag(L) @ Q.T,
        ret_from_gt_np=frontend_Q @ np.diag(frontend_L) @ frontend_Q.T,
        atol=1e-2,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )


# qr
@handle_frontend_test(
    fn_tree="jax.lax.linalg.qr",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        min_value=2,
        max_value=5,
    ),
    mode=st.sampled_from((True, False)),
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
        x=np.asarray(x[0], dtype[0]),
        full_matrices=mode,
    )


# svd
# TODO: implement proper drawing of index parameter and implement subset_by_index
# and to resolve groundtruth's significant inaccuracy
@handle_frontend_test(
    fn_tree="jax.lax.linalg.svd",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: (x, x)),
    ),
    full_matrices=st.booleans(),
    compute_uv=st.booleans(),
    index=st.one_of(
        st.none()
        # , st.tuples(st.integers(min_value=0, max_value=3),
        # st.integers(min_value=3, max_value=5))
    ),
    test_with_out=st.just(False),
)
def test_jax_svd(
    *,
    dtype_x,
    full_matrices,
    compute_uv,
    index,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
        subset_by_index=index,
    )
    if not compute_uv:
        if backend_fw == "torch":
            ret = ret.detach()
        assert_all_close(
            ret_np=np.asarray(frontend_ret, dtype=np.dtype(getattr(np, dtype[0]))),
            ret_from_gt_np=np.asarray(ret),
            rtol=1e-3,
            backend=backend_fw,
            ground_truth_backend=frontend,
        )
    else:
        if backend_fw == "torch":
            ret = [x.detach() for x in ret]
        ret = [np.asarray(x) for x in ret]
        frontend_ret = [
            np.asarray(x, dtype=np.dtype(getattr(np, dtype[0]))) for x in frontend_ret
        ]
        u, s, v = ret
        frontend_u, frontend_s, frontend_v = frontend_ret
        if not full_matrices:
            helpers.assert_all_close(
                ret_np=frontend_u @ np.diag(frontend_s) @ frontend_v.T,
                ret_from_gt_np=u @ np.diag(s) @ v.T,
                rtol=1e-3,
                backend=backend_fw,
                ground_truth_backend=frontend,
            )
        else:
            helpers.assert_all_close(
                ret_np=frontend_u[..., : frontend_s.shape[0]]
                @ np.diag(frontend_s)
                @ frontend_v.T,
                ret_from_gt_np=u[..., : s.shape[0]] @ np.diag(s) @ v.T,
                rtol=1e-3,
                backend=backend_fw,
                ground_truth_backend=frontend,
            )
