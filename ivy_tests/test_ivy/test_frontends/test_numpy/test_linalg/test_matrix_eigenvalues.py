# global
import sys
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import (
    handle_frontend_test,
    assert_all_close,
    BackendHandler,
)
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
)


# eig
@handle_frontend_test(
    fn_tree="numpy.linalg.eig",
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
    test_with_out=st.just(False),
)
def test_numpy_eig(
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    x = np.array(x[0], dtype=dtype[0])
    """Make symmetric positive-definite since ivy does not support complex data
    dtypes currently."""
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

    with BackendHandler.update_backend(backend_fw) as ivy_backend:
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
    fn_tree="numpy.linalg.eigh",
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
    UPLO=st.sampled_from(("L", "U")),
    test_with_out=st.just(False),
)
def test_numpy_eigh(
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
    )
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
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


@handle_frontend_test(
    fn_tree="numpy.linalg.eigvals",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=4).map(lambda x: (x, x)),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    test_with_out=st.just(False),
)
def test_numpy_eigvals(
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
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
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        ret = np.sort(
            np.array([ivy_backend.to_numpy(x).astype(np.float128) for x in ret])
        )
        frontend_ret = np.sort(np.array([x.astype(np.float128) for x in frontend_ret]))
        assert_all_close(
            ret_np=ret,
            ret_from_gt_np=frontend_ret,
            backend=backend_fw,
            ground_truth_backend=frontend,
            atol=1e-2,
            rtol=1e-2,
        )


# eigvalsh
@handle_frontend_test(
    fn_tree="numpy.linalg.eigvalsh",
    x=_get_dtype_and_matrix(symmetric=True),
    UPLO=st.sampled_from(["L", "U"]),
)
def test_numpy_eigvalsh(
    x,
    UPLO,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs = x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        a=xs,
        UPLO=UPLO,
    )
