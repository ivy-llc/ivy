# global
import ivy
from hypothesis import strategies as st, assume
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close
from ivy_tests.test_ivy.helpers import handle_frontend_test, matrix_is_stable
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
)

from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_linalg import (
    _get_second_matrix,
    _get_cholesky_matrix,
)

from ivy_tests.test_ivy.test_frontends.test_torch.test_blas_and_lapack_ops import (
    _get_dtype_input_and_mat_vec,
)


# --- Helpers --- #
# --------------- #


@st.composite
def _dtype_values_axis(draw):
    dtype_and_values = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=2,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=5,
            min_value=0.1,
            max_value=1000.0,
        )
    )

    dtype, x = dtype_and_values
    x = x[0]
    r = len(x.shape)

    valid_axes = [None]

    for i in range(-r, r):
        valid_axes.append(i)
        for j in range(-r, r):
            if i != j and abs(i - j) != r:
                valid_axes.append([i, j])

    axis = draw(st.sampled_from(valid_axes))

    p_list = ["fro", 1, 2, ivy.inf, -ivy.inf]
    if isinstance(axis, list) and len(axis) == 2:
        p = draw(
            st.one_of(
                st.sampled_from(p_list),
                st.floats(min_value=1.0, max_value=10.0, allow_infinity=False),
            )
        )
    else:
        p = draw(
            st.one_of(
                st.sampled_from(p_list + [0]),
                st.floats(min_value=1.0, max_value=10.0, allow_infinity=False),
            )
        )

    return dtype, x, axis, p


# cond
@st.composite
def _get_dtype_and_matrix_non_singular(draw, dtypes):
    while True:
        matrix = draw(
            helpers.dtype_and_values(
                available_dtypes=dtypes,
                min_value=-10,
                max_value=10,
                min_num_dims=2,
                max_num_dims=2,
                min_dim_size=1,
                max_dim_size=5,
                shape=st.tuples(st.integers(1, 5), st.integers(1, 5)).filter(
                    lambda x: x[0] == x[1]
                ),
                allow_inf=False,
                allow_nan=False,
            )
        )
        if np.linalg.det(matrix[1][0]) != 0:
            break

    return matrix[0], matrix[1]


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


# cholesky_solve
@st.composite
def _get_paddle_cholesky_matrix(draw):
    input_dtype, spd_chol = draw(_get_cholesky_matrix())
    probability = draw(st.floats(min_value=0, max_value=1))
    if probability > 0.5:
        spd_chol = spd_chol.T  # randomly transpose the matrix
    return input_dtype, spd_chol


# transpose
@st.composite
def _transpose_helper(draw):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=4,
            min_dim_size=2,
            max_dim_size=3,
            ret_shape=True,
        )
    )
    perm = draw(st.permutations([i for i in range(len(shape))]))
    return dtype, x, perm


# Helpers #
# ------ #


@st.composite
def dtype_value1_value2_axis(
    draw,
    available_dtypes,
    abs_smallest_val=None,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
    specific_dim_size=3,
    large_abs_safety_factor=4,
    small_abs_safety_factor=4,
    safety_factor_scale="log",
):
    # For cross product, a dim with size 3 is required
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    axis = draw(helpers.ints(min_value=0, max_value=len(shape)))
    # make sure there is a dim with specific dim size
    shape = list(shape)
    shape = shape[:axis] + [specific_dim_size] + shape[axis:]
    shape = tuple(shape)

    dtype = draw(st.sampled_from(draw(available_dtypes)))

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    abs_smallest_val=abs_smallest_val,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                    large_abs_safety_factor=large_abs_safety_factor,
                    small_abs_safety_factor=small_abs_safety_factor,
                    safety_factor_scale=safety_factor_scale,
                )
            )
        )

    value1, value2 = values[0], values[1]
    return [dtype], value1, value2, axis


# --- Main --- #
# ------------ #


# bincount
@handle_frontend_test(
    fn_tree="paddle.bincount",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1,
        max_value=2,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=1,
            ),
            key="a_s_d",
        ),
    ),
    test_with_out=st.just(False),
)
def test_paddle_bincount(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        weights=None,
        minlength=0,
    )


# bmm
@handle_frontend_test(
    fn_tree="paddle.bmm",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=(3, 3, 3),
        num_arrays=2,
        shared_dtype=True,
        min_value=-10,
        max_value=10,
    ),
    test_with_out=st.just(False),
)
def test_paddle_bmm(
    *,
    dtype_x,
    frontend,
    test_flags,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# cholesky
@handle_frontend_test(
    fn_tree="paddle.cholesky",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: (x, x)),
    ),
    upper=st.booleans(),
)
def test_paddle_cholesky(
    dtype_and_x,
    upper,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    x = x[0]
    x = np.matmul(x.T, x) + np.identity(x.shape[0])  # make symmetric positive-definite

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        upper=upper,
    )


@handle_frontend_test(
    fn_tree="paddle.linalg.cholesky_solve",
    x=_get_second_matrix(),
    y=_get_paddle_cholesky_matrix(),
    test_with_out=st.just(False),
)
def test_paddle_cholesky_solve(
    *,
    x,
    y,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype1, x1 = x
    input_dtype2, x2 = y
    helpers.test_frontend_function(
        input_dtypes=[input_dtype1, input_dtype2],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        x=x1,
        y=x2,
        upper=np.array_equal(x2, np.triu(x2)),  # check whether the matrix is upper
    )


@handle_frontend_test(
    fn_tree="paddle.linalg.cond",
    dtype_and_x=_get_dtype_and_matrix_non_singular(dtypes=["float32", "float64"]),
    p=st.sampled_from([None, "fro", "nuc", np.inf, -np.inf, 1, -1, 2, -2]),
    test_with_out=st.just(False),
)
def test_paddle_cond(
    *, dtype_and_x, p, on_device, fn_tree, frontend, test_flags, backend_fw
):
    dtype, x = dtype_and_x

    assume(matrix_is_stable(x[0]))

    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        x=x[0],
        rtol=1e-5,
        atol=1e-5,
        p=p,
    )


# cov
@handle_frontend_test(
    fn_tree="paddle.linalg.cov",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        max_num_dims=2,
    ),
    rowvar=st.booleans(),
    ddof=st.booleans(),
)
def test_paddle_cov(
    *,
    dtype_and_x,
    rowvar,
    ddof,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        x=x,
        rowvar=rowvar,
        ddof=ddof,
    )


# Tests #
# ----- #


# cross
@handle_frontend_test(
    fn_tree="paddle.cross",
    dtype_x_y_axis=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-1e5,
        max_value=1e5,
        abs_smallest_val=0.01,
        safety_factor_scale="log",
    ),
)
def test_paddle_cross(
    *,
    dtype_x_y_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x, y, axis = dtype_x_y_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        y=y,
        axis=axis,
        atol=1e-4,
        rtol=1e-4,
    )


@handle_frontend_test(
    fn_tree="paddle.dist",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    p=helpers.floats(min_value=1.0, max_value=10.0),
)
def test_paddle_dist(
    *,
    dtype_and_input,
    p,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
        p=p,
    )


# dot
@handle_frontend_test(
    fn_tree="paddle.dot",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_paddle_dot(
    *,
    dtype_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# eig
@handle_frontend_test(
    fn_tree="paddle.linalg.eig",
    dtype_and_input=_get_dtype_and_square_matrix(real_and_complex_only=True),
    test_with_out=st.just(False),
)
def test_paddle_eig(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_input
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    if x.dtype == ivy.float32:
        x = x.astype("float64")
        input_dtype = [ivy.float64]
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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

    l, v = ret  # noqa: E741
    front_l, front_v = frontend_ret

    assert_all_close(
        ret_np=v @ np.diag(l) @ v.T,
        ret_from_gt_np=front_v @ np.diag(front_l) @ front_v.T,
        rtol=1e-2,
        atol=1e-2,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )


# eigh
@handle_frontend_test(
    fn_tree="paddle.linalg.eigh",
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
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_input
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    if x.dtype == ivy.float32:
        x = x.astype("float64")
        input_dtype = [ivy.float64]
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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

    l, v = ret  # noqa: E741
    front_l, front_v = frontend_ret

    assert_all_close(
        ret_np=v @ np.diag(l) @ v.T,
        ret_from_gt_np=front_v @ np.diag(front_l) @ front_v.T,
        rtol=1e-2,
        atol=1e-2,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )


# eigvals
@handle_frontend_test(
    fn_tree="paddle.linalg.eigvals",
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
    backend_fw,
):
    dtype, x = dtype_x
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
    )


# eigvalsh
@handle_frontend_test(
    fn_tree="paddle.eigvalsh",
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
    backend_fw,
):
    dtype, x = dtype_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x,
        UPLO=UPLO,
    )


# diagonal
@handle_frontend_test(
    fn_tree="paddle.diagonal",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    axis_and_offset=helpers.dims_and_offset(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape")
    ),
)
def test_paddle_linalg_diagonal(
    dtype_and_values,
    axis_and_offset,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, value = dtype_and_values
    axis1, axis2, offset = axis_and_offset
    input = value[0]
    num_dims = len(np.shape(input))
    assume(axis1 != axis2)
    if axis1 < 0:
        assume(axis1 + num_dims != axis2)
    if axis2 < 0:
        assume(axis1 != axis2 + num_dims)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        on_device=on_device,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=input,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


@handle_frontend_test(
    fn_tree="paddle.lu_unpack",
    dtype_x=_get_dtype_and_square_matrix(real_and_complex_only=True),
    p=st.lists(st.floats(1, 5), max_size=5),
    unpack_datas=st.booleans(),
    unpack_pivots=st.booleans(),
)
def test_paddle_lu_unpack(
    *,
    dtype_x,
    p,
    unpack_datas,
    unpack_pivots,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_x
    x = np.array(x[0], dtype=dtype[0])
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        lu_data=x,
        lu_pivots=p,
        unpack_datas=unpack_datas,
        unpack_pivots=unpack_pivots,
        rtol=1e-03,
        atol=1e-03,
    )


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
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
        transpose_x=transpose_x,
        transpose_y=transpose_y,
    )


# matrix_power
@handle_frontend_test(
    fn_tree="paddle.linalg.matrix_power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        shape=helpers.ints(min_value=2, max_value=8).map(lambda x: (x, x)),
    ),
    n=helpers.ints(min_value=1, max_value=8),
    test_with_out=st.just(False),
)
def test_paddle_matrix_power(
    dtype_and_x,
    n,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        n=n,
    )


# mv
@handle_frontend_test(
    fn_tree="paddle.mv",
    dtype_mat_vec=_get_dtype_input_and_mat_vec(),
    test_with_out=st.just(False),
)
def test_paddle_mv(
    dtype_mat_vec,
    frontend,
    test_flags,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, mat, vec = dtype_mat_vec
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=mat,
        vec=vec,
    )


# norm
@handle_frontend_test(
    fn_tree="paddle.norm",
    dtype_values_axis=_dtype_values_axis(),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
)
def test_paddle_norm(
    dtype_values_axis,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x, axis, p = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        p=p,
        axis=axis,
        keepdim=keepdims,
        atol=1e-1,
        rtol=1e-1,
    )


# pinv
@handle_frontend_test(
    fn_tree="paddle.linalg.pinv",
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
    backend_fw,
    on_device,
):
    # TODO: paddle returns nan for all values if the input
    # matrix has the same value at all indices e.g.
    # [[2., 2.], [2., 2.]] would return [[nan, nan], [nan, nan]],
    # causing the tests to fail for other backends.
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        x=x[0],
        rcond=rcond,
    )


# qr
@handle_frontend_test(
    fn_tree="paddle.linalg.qr",
    dtype_and_x=_get_dtype_and_matrix(),
    mode=st.sampled_from(("reduced", "complete")),
    test_with_out=st.just(False),
)
def test_paddle_qr(
    dtype_and_x,
    mode,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    assume(matrix_is_stable(x[0]))
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        x=x[0],
        mode=mode,
    )


# solve
@handle_frontend_test(
    fn_tree="paddle.linalg.solve",
    x=helpers.get_first_solve_batch_matrix(),
    y=helpers.get_second_solve_batch_matrix(),
    test_with_out=st.just(False),
)
def test_paddle_solve(
    *,
    x,
    y,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype1, x1, _ = x
    input_dtype2, x2, _ = y
    helpers.test_frontend_function(
        input_dtypes=[input_dtype1, input_dtype2],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        x=x1,
        y=x2,
    )


@handle_frontend_test(
    fn_tree="paddle.transpose",
    dtype_and_x_perm=_transpose_helper(),
    test_with_out=st.just(False),
)
def test_paddle_transpose(
    dtype_and_x_perm,
    frontend,
    test_flags,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x, perm = dtype_and_x_perm
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        perm=perm,
    )
