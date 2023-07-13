# global
import sys
import numpy as np
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
)


# cholesky
@handle_frontend_test(
    fn_tree="numpy.linalg.cholesky",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(x[1][0]) != 0
    ),
    test_with_out=st.just(False),
)
def test_numpy_cholesky(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    x = x[0]
    x = (
        np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    )  # make symmetric positive-definite
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        a=x,
    )


# qr
@handle_frontend_test(
    fn_tree="numpy.linalg.qr",
    dtype_and_x=_get_dtype_and_matrix(),
    mode=st.sampled_from(("reduced", "complete")),
    test_with_out=st.just(False),
)
def test_numpy_qr(
    dtype_and_x,
    mode,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        a=x,
        mode=mode,
    )


# svd
@handle_frontend_test(
    fn_tree="numpy.linalg.svd",
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
def test_numpy_svd(
    dtype_and_x,
    full_matrices,
    compute_uv,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    x = x[0]
    x = (
        np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    )  # make symmetric positive-definite
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        test_values=False,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
    )
    for u, v in zip(ret, ret_gt):
        u = ivy.to_numpy(ivy.abs(u))
        v = ivy.to_numpy(ivy.abs(v))
        helpers.value_test(ret_np_flat=u, ret_np_from_gt_flat=v, rtol=1e-04, atol=1e-04)
