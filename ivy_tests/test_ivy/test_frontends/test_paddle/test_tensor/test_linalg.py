# global
import ivy
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Helpers #
# ------ #


@st.composite
def _get_dtype_input_and_vectors(draw):
    dim_size = draw(helpers.ints(min_value=1, max_value=2))
    dtype = draw(helpers.get_dtypes("float"))
    if dim_size == 1:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
    else:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
    return dtype, vec1, vec2


@st.composite
def _get_dtype_and_square_matrix(draw, real_and_complex_only=False):
    if real_and_complex_only:
        dtype = [
            draw(st.sampled_from(["float32", "float64", "complex64", "complex128"]))
        ]
    else:
        dtype = draw(helpers.get_dtypes("valid"))
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    mat = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=0, max_value=10
        )
    )
    return dtype, mat


# Tests #
# ----- #


# matmul
@handle_frontend_test(
    fn_tree="paddle.matmul",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=(3, 3),
        num_arrays=2,
        shared_dtype=True,
        min_value=-10,
        max_value=10,
    ),
    aliases=["paddle.tensor.linalg.matmul"],
    transpose_x=st.booleans(),
    transpose_y=st.booleans(),
    test_with_out=st.just(False),
)
def test_paddle_matmul(
    *,
    dtype_x,
    transpose_x,
    transpose_y,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
        transpose_x=transpose_x,
        transpose_y=transpose_y,
    )


# eig
@handle_frontend_test(
    fn_tree="paddle.tensor.linalg.eig",
    dtype_and_input=_get_dtype_and_square_matrix(real_and_complex_only=True),
    test_with_out=st.just(False),
)
def test_paddle_eig(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    if x.dtype == ivy.float32:
        x = x.astype("float64")
        input_dtype = [ivy.float64]
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        atol=1e-4,
        x=x,
    )
    ret = [ivy.to_numpy(x).astype("float64") for x in ret]
    frontend_ret = [np.asarray(x, dtype=np.float64) for x in frontend_ret]

    l, v = ret
    front_l, front_v = frontend_ret

    assert_all_close(
        ret_np=v @ np.diag(l) @ v.T,
        ret_from_gt_np=front_v @ np.diag(front_l) @ front_v.T,
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
    )


# eigvals
@handle_frontend_test(
    fn_tree="paddle.tensor.linalg.eigvals",
    dtype_x=_get_dtype_and_square_matrix(real_and_complex_only=True),
    test_with_out=st.just(False),
)
def test_paddle_eigvals(
    *,
    dtype_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_x
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
    )


# eigvalsh
@handle_frontend_test(
    fn_tree="paddle.tensor.linalg.eigvalsh",
    dtype_x=_get_dtype_and_square_matrix(real_and_complex_only=True),
    UPLO=st.sampled_from(("L", "U")),
    test_with_out=st.just(False),
)
def test_paddle_eigvalsh(
    *,
    dtype_x,
    UPLO,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x,
        UPLO=UPLO,
    )


# eigh
@handle_frontend_test(
    fn_tree="paddle.tensor.linalg.eigh",
    dtype_and_input=_get_dtype_and_square_matrix(real_and_complex_only=True),
    UPLO=st.sampled_from(("L", "U")),
    test_with_out=st.just(False),
)
def test_paddle_eigh(
    *,
    dtype_and_input,
    UPLO,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    if x.dtype == ivy.float32:
        x = x.astype("float64")
        input_dtype = [ivy.float64]
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        atol=1e-4,
        x=x,
        UPLO=UPLO,
    )
    ret = [ivy.to_numpy(x).astype("float64") for x in ret]
    frontend_ret = [np.asarray(x, dtype=np.float64) for x in frontend_ret]

    l, v = ret
    front_l, front_v = frontend_ret

    assert_all_close(
        ret_np=v @ np.diag(l) @ v.T,
        ret_from_gt_np=front_v @ np.diag(front_l) @ front_v.T,
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
    )


# pinv
@handle_frontend_test(
    fn_tree="paddle.tensor.linalg.pinv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        min_value=3,
        max_value=10,
        large_abs_safety_factor=128,
        safety_factor_scale="log",
    ),
    rcond=st.floats(1e-5, 1e-3),
    test_with_out=st.just(False),
)
def test_paddle_pinv(
    dtype_and_x,
    rcond,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    # TODO: paddle returns nan for all values if the input
    # matrix has the same value at all indices e.g.
    # [[2., 2.], [2., 2.]] would return [[nan, nan], [nan, nan]],
    # causing the tests to fail for other backends.
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        x=x[0],
        rcond=rcond,
    )
