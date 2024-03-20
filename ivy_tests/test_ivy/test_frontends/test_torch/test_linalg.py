# global
import math
import sys
import numpy as np
from hypothesis import strategies as st, assume, settings, HealthCheck

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_torch.test_miscellaneous_ops import (
    dtype_value1_value2_axis,
)
from ivy_tests.test_ivy.helpers.hypothesis_helpers.general_helpers import (
    matrix_is_stable,
)
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import _matrix_rank_helper


# --- Helpers --- #
# --------------- #


@st.composite
def _generate_multi_dot_dtype_and_arrays(draw):
    input_dtype = [draw(st.sampled_from(draw(helpers.get_dtypes("valid"))))]
    matrices_dims = draw(
        st.lists(st.integers(min_value=2, max_value=10), min_size=4, max_size=4)
    )
    shape_1 = (matrices_dims[0], matrices_dims[1])
    shape_2 = (matrices_dims[1], matrices_dims[2])
    shape_3 = (matrices_dims[2], matrices_dims[3])

    matrix_1 = draw(
        helpers.dtype_and_values(
            shape=shape_1,
            dtype=input_dtype,
            min_value=-10,
            max_value=10,
        )
    )
    matrix_2 = draw(
        helpers.dtype_and_values(
            shape=shape_2,
            dtype=input_dtype,
            min_value=-10,
            max_value=10,
        )
    )
    matrix_3 = draw(
        helpers.dtype_and_values(
            shape=shape_3,
            dtype=input_dtype,
            min_value=-10,
            max_value=10,
        )
    )

    return input_dtype, [matrix_1[1][0], matrix_2[1][0], matrix_3[1][0]]


@st.composite
def _get_axis_and_p(draw):
    p = draw(st.sampled_from(["fro", "nuc", 1, 2, -1, -2, float("inf"), -float("inf")]))
    if p in ["fro", "nuc"]:
        max_axes_size = 2
        min_axes_size = 2
    else:
        min_axes_size = 1
        max_axes_size = 2
    x_dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=2,
            valid_axis=True,
            min_value=-1e04,
            max_value=1e04,
            min_axes_size=min_axes_size,
            max_axes_size=max_axes_size,
            large_abs_safety_factor=2,
            safety_factor_scale="log",
        )
    )
    axis = axis[0] if isinstance(axis, tuple) and len(axis) == 1 else axis
    # ToDo: fix the castable dtype helper. Right now using `dtype` causes errors
    #  dtype should be real for real inputs, but got ComplexDouble
    x_dtype, values, dtype = draw(
        helpers.get_castable_dtype(
            draw(helpers.get_dtypes("valid")), x_dtype[0], values[0]
        )
    )
    return p, x_dtype, values, axis, x_dtype


# helpers
@st.composite
def _get_dtype_and_matrix(
    draw, dtype="valid", square=False, invertible=False, batch=False
):
    if batch:
        arbitrary_dims = draw(helpers.get_shape(max_dim_size=3))
    else:
        arbitrary_dims = []
    if square:
        random_size = draw(st.integers(1, 5))
        shape = (*arbitrary_dims, random_size, random_size)
    else:
        shape = (*arbitrary_dims, draw(st.integers(1, 5)), draw(st.integers(1, 5)))
    ret = helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(dtype, full=True),
        min_value=-10,
        max_value=10,
        abs_smallest_val=1e-04,
        shape=shape,
    )
    if invertible:
        ret = ret.filter(
            lambda x: np.all(np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon)
        )
    return draw(ret)


@st.composite
def _get_dtype_and_symmetrix_matrix(draw):
    input_dtype = draw(st.shared(st.sampled_from(draw(helpers.get_dtypes("valid")))))
    random_size = draw(helpers.ints(min_value=2, max_value=4))
    batch_shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=3))
    num_independnt_vals = int((random_size**2) / 2 + random_size / 2)
    array_vals_flat = np.array(
        draw(
            helpers.array_values(
                dtype=input_dtype,
                shape=tuple(list(batch_shape) + [num_independnt_vals]),
                min_value=2,
                max_value=5,
            )
        )
    )
    array_vals = np.zeros(batch_shape + (random_size, random_size))
    c = 0
    for i in range(random_size):
        for j in range(random_size):
            if j < i:
                continue
            array_vals[..., i, j] = array_vals_flat[..., c]
            array_vals[..., j, i] = array_vals_flat[..., c]
            c += 1
    return [input_dtype], array_vals


# tensorsolve
@st.composite
def _get_solve_matrices(draw):
    # batch_shape, random_size, shared

    # float16 causes a crash when filtering out matrices
    # for which `np.linalg.cond` is large.
    input_dtype_strategy = st.shared(
        st.sampled_from(draw(helpers.get_dtypes("valid"))),
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
        ).filter(
            lambda x: np.linalg.cond(x.reshape((dim**2, dim**2)))
            < 1 / sys.float_info.epsilon
        )
    )

    second_matrix = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=(dim, dim),
            min_value=1.2,
            max_value=3,
        ).filter(
            lambda x: np.linalg.cond(x.reshape((dim, dim))) < 1 / sys.float_info.epsilon
        )
    )

    return input_dtype, first_matrix, second_matrix


# tensorinv
@st.composite
def _tensorinv_helper(draw):
    def factors(x):
        result = [
            1,
        ]
        i = 2
        while i * i <= x:
            if x % i == 0:
                result.append(i)
                if x // i != i:
                    result.append(x // i)
            i += 1
        result.append(x)
        return np.array(result)

    ind = draw(helpers.ints(min_value=1, max_value=6))
    product_half = draw(helpers.ints(min_value=2, max_value=25))
    factors_list = factors(product_half)
    shape = ()
    while len(shape) < ind and ind > 2:
        while np.prod(shape) < product_half:
            a = factors_list[np.random.randint(len(factors_list))]
            shape += (a,)
        if np.prod(shape) > product_half or len(shape) > ind:
            shape = ()
        while len(shape) < ind and shape != ():
            shape += (1,)
        if np.prod(shape) == product_half:
            shape += shape[::-1]
            break
    if ind == 1 and shape == ():
        shape += (product_half, product_half)
    if ind == 2 and shape == ():
        shape += (1, product_half, product_half, 1)
    shape_cor = ()
    for i in shape:
        shape_cor += (int(i),)
    shape_draw = (product_half, product_half)
    dtype, input = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape_draw,
        ).filter(lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon)
    )
    input[0] = input[0].reshape(shape_cor)
    return dtype, input[0], ind


# vander
@st.composite
def _vander_helper(draw):
    # generate input matrix of shape (*, n) and where '*' is one or more
    # batch dimensions
    N = draw(helpers.ints(min_value=2, max_value=5))
    if draw(st.floats(min_value=0, max_value=1.0)) < 0.5:
        N = None

    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x = draw(
        helpers.dtype_and_values(
            available_dtypes=draw(helpers.get_dtypes("valid")),
            shape=shape,
            min_value=-10,
            max_value=10,
        )
    )

    return *x, N


# --- Main --- #
# ------------ #


@handle_frontend_test(
    fn_tree="torch.linalg.lu_solve",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=helpers.get_shape(
            min_num_dims=2, max_num_dims=2, min_dim_size=2, max_dim_size=2
        ),
        num_arrays=2,
        shared_dtype=True,
    ).filter(lambda x: helpers.matrix_is_stable(x[1][0], cond_limit=10)),
)
def test_lu_solve(dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, arr = dtype_x
    A, B = arr[0], arr[1]
    ivy.set_backend(backend_fw)
    lu_ = ivy.lu_factor(A)
    lu, p = lu_.LU, lu_.p
    X, X_gt = helpers.test_frontend_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        lu=lu,
        p=p,
        b=B,
        test_values=False,
    )
    assert np.allclose(A @ X, B)


@handle_frontend_test(
    fn_tree="torch.linalg.cholesky",
    aliases=["torch.cholesky"],
    dtype_and_x=_get_dtype_and_matrix(square=True),
    upper=st.booleans(),
)
def test_torch_cholesky(
    *,
    dtype_and_x,
    upper,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    x = np.matmul(x.T, x) + np.identity(x.shape[0])  # make symmetric positive-definite

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=x,
        upper=upper,
    )


@handle_frontend_test(
    fn_tree="torch.linalg.cholesky_ex",
    dtype_and_x=_get_dtype_and_matrix(square=True, batch=True),
    upper=st.booleans(),
)
def test_torch_cholesky_ex(
    *,
    dtype_and_x,
    upper,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    x = np.matmul(x.T, x) + np.identity(x.shape[0])  # make symmetric positive-definite

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=x,
        upper=upper,
    )


@handle_frontend_test(
    fn_tree="torch.linalg.cond",
    dtype_and_x=_get_dtype_and_matrix(square=True, invertible=True, batch=True),
    p=st.sampled_from([None, "fro", "nuc", np.inf, -np.inf, 1, -1, 2, -2]),
)
def test_torch_cond(
    *, dtype_and_x, p, on_device, fn_tree, frontend, backend_fw, test_flags
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        input=x[0],
        rtol=1e-2,
        atol=1e-3,
        p=p,
    )


# cross
@handle_frontend_test(
    fn_tree="torch.linalg.cross",
    dtype_input_other_dim=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-1e3,
        max_value=1e3,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_torch_cross(
    dtype_input_other_dim,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    dtype, input, other, dim = dtype_input_other_dim
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        rtol=1e-2,
        atol=1e-3,
        input=input,
        other=other,
        dim=dim,
    )


# det
@handle_frontend_test(
    fn_tree="torch.linalg.det",
    aliases=["torch.det"],
    dtype_and_x=_get_dtype_and_matrix(square=True, batch=True),
)
def test_torch_det(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    test_flags.num_positional_args = len(x)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        A=x[0],
    )


@handle_frontend_test(
    fn_tree="torch.diag_embed",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=2), key="shape"),
    ),
    dims_and_offsets=helpers.dims_and_offset(
        shape=st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=2), key="shape"),
        ensure_dim_unique=True,
    ),
)
@settings(suppress_health_check=list(HealthCheck))
def test_torch_diag_embed(
    *,
    dtype_and_values,
    dims_and_offsets,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, value = dtype_and_values
    dim1, dim2, offset = dims_and_offsets
    num_of_dims = len(np.shape(value[0]))
    if dim1 < 0:
        assume(dim1 + num_of_dims != dim2)
    if dim2 < 0:
        assume(dim1 != dim2 + num_of_dims)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        offset=offset,
        dim1=dim1,
        dim2=dim2,
    )


# eig
# TODO: Test for all valid dtypes once ivy.eig supports complex data types
@handle_frontend_test(
    fn_tree="torch.linalg.eig",
    dtype_and_input=_get_dtype_and_matrix(dtype="float", square=True),
)
def test_torch_eig(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_input
    x = np.asarray(x[0], dtype=input_dtype[0])
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
        input=x,
    )
    ret = [ivy.to_numpy(x).astype("float64") for x in ret]
    frontend_ret = [np.asarray(x, dtype=np.float64) for x in frontend_ret]

    l, v = ret  # noqa: E741
    front_l, front_v = frontend_ret

    assert_all_close(
        ret_np=v @ np.diag(l) @ np.linalg.inv(v),
        ret_from_gt_np=front_v @ np.diag(front_l) @ np.linalg.inv(front_v),
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
        backend=backend_fw,
    )


@handle_frontend_test(
    fn_tree="torch.linalg.eigh",
    dtype_and_x=_get_dtype_and_matrix(dtype="valid", square=True, invertible=True),
    UPLO=st.sampled_from(("L", "U")),
)
def test_torch_eigh(
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
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    L, Q = ret
    frontend_L, frontend_Q = frontend_ret

    assert_all_close(
        ret_np=Q @ np.diag(L) @ Q.T,
        ret_from_gt_np=frontend_Q @ np.diag(frontend_L) @ frontend_Q.T,
        atol=1e-02,
        backend=backend_fw,
    )


# eigvals
@handle_frontend_test(
    fn_tree="torch.linalg.eigvals",
    dtype_x=_get_dtype_and_matrix(square=True),
)
def test_torch_eigvals(
    *,
    dtype_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_x

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        test_values=False,
    )
    """In "ret" we have out eigenvalues calculated with our backend and in
    "frontend_ret" are our eigenvalues calculated with the specified
    frontend."""

    """
    Depending on the chosen framework there may be small differences between our
    extremely small or big eigenvalues (eg: -3.62831993e-33+0.j(numpy)
    vs -1.9478e-32+0.j(PyTorch)).
    Important is that both are very very close to zero, indicating a
    small value(very close to 0) either way.

    To asses the correctness of our calculated eigenvalues for our initial matrix
    we sort both numpy arrays and call assert_all_close on their modulus.
    """

    """
    Supports input of float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices, and if A is a batch of matrices then the
    output has the same batch dimension
    """

    frontend_ret = np.asarray(frontend_ret[0])
    frontend_ret = np.sort(frontend_ret)
    frontend_ret_modulus = np.zeros(len(frontend_ret), dtype=np.float64)
    for i in range(len(frontend_ret)):
        frontend_ret_modulus[i] = math.sqrt(
            math.pow(frontend_ret[i].real, 2) + math.pow(frontend_ret[i].imag, 2)
        )

    ret = ivy.to_numpy(ret).astype(str(frontend_ret.dtype))
    ret = np.sort(ret)
    ret_modulus = np.zeros(len(ret), dtype=np.float64)
    for i in range(len(ret)):
        ret_modulus[i] = math.sqrt(math.pow(ret[i].real, 2) + math.pow(ret[i].imag, 2))

    assert_all_close(
        ret_np=ret_modulus,
        ret_from_gt_np=frontend_ret_modulus,
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
        backend=backend_fw,
    )


# eigvalsh
@handle_frontend_test(
    fn_tree="torch.linalg.eigvalsh",
    dtype_x=_get_dtype_and_symmetrix_matrix(),
    UPLO=st.sampled_from(("L", "U")),
)
def test_torch_eigvalsh(
    *,
    dtype_x,
    UPLO,
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
        input=x,
        UPLO=UPLO,
        atol=1e-4,
        rtol=1e-3,
    )


# inv
@handle_frontend_test(
    fn_tree="torch.linalg.inv",
    aliases=["torch.inverse"],
    dtype_and_x=_get_dtype_and_matrix(square=True, invertible=True, batch=True),
)
def test_torch_inv(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    test_flags.num_positional_args = 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        atol=1e-03,
        A=x[0],
    )


# inv_ex
# TODO: Test for singular matrices
@handle_frontend_test(
    fn_tree="torch.linalg.inv_ex",
    dtype_and_x=_get_dtype_and_matrix(square=True, invertible=True, batch=True),
)
def test_torch_inv_ex(
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
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        atol=1e-02,
        A=x[0],
    )


# lu_factor
@handle_frontend_test(
    fn_tree="torch.linalg.lu_factor",
    input_dtype_and_input=_get_dtype_and_matrix(batch=True),
)
def test_torch_lu_factor(
    *,
    input_dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input = input_dtype_and_input
    ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        atol=1e-02,
        A=input[0],
        test_values=False,
    )
    ret_f, ret_gt = ret
    LU, p = ret_f.LU, ret_f.p
    L = np.tril(LU, -1) + np.eye(LU.shape[0])
    U = np.triu(LU)
    P = np.eye(LU.shape[0])[p]
    assert np.allclose(L @ U, P @ input[0])


@handle_frontend_test(
    fn_tree="torch.linalg.lu_factor_ex",
    input_dtype_and_input=_get_dtype_and_matrix(
        batch=True, square=True, invertible=True
    ),
)
def test_torch_lu_factor_ex(
    *,
    input_dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input = input_dtype_and_input
    ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        atol=1e-02,
        A=input[0],
        check_errors=False,
        test_values=False,
    )
    ret_f, ret_gt = ret
    ret_f_matrix, ret_f_info = ret_f
    if ret_f_info == 0:
        (
            LU,
            p,
        ) = (
            ret_f_matrix.LU,
            ret_f_matrix.p,
        )
        L = np.tril(LU, -1) + np.eye(LU.shape[0])
        U = np.triu(LU)
        P = np.eye(LU.shape[0])[p]
        assert np.allclose(L @ U, P @ input[0])


@handle_frontend_test(
    fn_tree="torch.linalg.matmul",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=(3, 3),
        num_arrays=2,
        shared_dtype=True,
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_matmul(
    *,
    dtype_x,
    frontend,
    fn_tree,
    on_device,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_flags=test_flags,
        input=x[0],
        other=x[1],
        rtol=1e-03,
        atol=1e-03,
    )


# matrix_exp
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_exp",
    dtype_and_x=_get_dtype_and_matrix(square=True, invertible=True),
)
def test_torch_matrix_exp(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    test_flags.num_positional_args = len(x)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        atol=1e-03,
        A=x[0],
    )


# matrix_norm
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_norm",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        min_axes_size=2,
        max_axes_size=2,
        max_value=10e4,
        min_value=-10e4,
        abs_smallest_val=10e-4,
        valid_axis=True,
        force_tuple_axis=True,
    ),
    ord=st.sampled_from(["fro", "nuc", np.inf, -np.inf, 1, -1, 2, -2]),
    keepdim=st.booleans(),
    dtypes=st.sampled_from((None, "16", "32", "64")),
)
def test_torch_matrix_norm(
    *,
    dtype_values_axis,
    ord,
    keepdim,
    frontend,
    dtypes,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_values_axis
    if dtypes is not None:
        # torch backend does not allow down-casting.
        if input_dtype[0] == "complex128":
            dtypes = input_dtype[0]
        else:
            dtypes = input_dtype[0][0:-2] + max([input_dtype[0][-2:], dtypes])

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        atol=1e-03,
        input=x[0],
        ord=ord,
        dim=axis,
        keepdim=keepdim,
        dtype=dtypes,
    )


# matrix_power
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_power",
    aliases=["torch.matrix_power"],
    dtype_and_x=_get_dtype_and_matrix(square=True, invertible=True, batch=True),
    n=helpers.ints(min_value=2, max_value=5),
)
def test_torch_matrix_power(
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
    test_flags.num_positional_args = len(x) + 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        A=x[0],
        n=n,
    )


# matrix_rank
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_rank",
    dtype_x_hermitian_atol_rtol=_matrix_rank_helper(),
)
def test_torch_matrix_rank(
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
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=0,
        atol=0,
        input=x,
        arg_rtol=rtol,
        arg_atol=atol,
        hermitian=hermitian,
    )


@handle_frontend_test(
    fn_tree="torch.linalg.multi_dot",
    dtype_x=_generate_multi_dot_dtype_and_arrays(),
)
def test_torch_multi_dot(
    dtype_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        test_values=True,
        tensors=x,
    )


@handle_frontend_test(
    fn_tree="torch.linalg.norm",
    args=_get_axis_and_p(),
    keepdim=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_norm(
    *,
    args,
    keepdim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    p, x_dtype, x, axis, dtype = args
    helpers.test_frontend_function(
        input_dtypes=[x_dtype],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-08,
        input=x,
        ord=p,
        dim=axis,
        keepdim=keepdim,
        dtype=dtype,
    )


# pinv
# TODO: add testing for hermitian
@handle_frontend_test(
    fn_tree="torch.linalg.pinv",
    dtype_and_input=_get_dtype_and_matrix(batch=True),
)
def test_torch_pinv(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        atol=1e-02,
        rtol=1e-02,
    )


# qr
@handle_frontend_test(
    fn_tree="torch.linalg.qr",
    dtype_and_input=_get_dtype_and_matrix(batch=True),
)
def test_torch_qr(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_input
    ivy.set_backend(backend_fw)
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        A=x[0],
        test_values=False,
    )
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    q, r = ret
    frontend_q, frontend_r = frontend_ret

    assert_all_close(
        ret_np=q @ r,
        ret_from_gt_np=frontend_q @ frontend_r,
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
        backend=backend_fw,
    )
    ivy.previous_backend()


# slogdet
@handle_frontend_test(
    fn_tree="torch.linalg.slogdet",
    aliases=["torch.slogdet"],
    dtype_and_x=_get_dtype_and_matrix(square=True, batch=True),
)
def test_torch_slogdet(
    *,
    dtype_and_x,
    fn_tree,
    frontend,
    on_device,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    test_flags.num_positional_args = len(x)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        A=x[0],
    )


# solve
@handle_frontend_test(
    fn_tree="torch.linalg.solve",
    A=helpers.get_first_solve_batch_matrix(),
    B=helpers.get_second_solve_batch_matrix(choose_side=True),
)
def test_torch_solve(
    *,
    A,
    B,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, A, _ = A
    _, B, left = B
    test_flags.num_positional_args = 2
    helpers.test_frontend_function(
        input_dtypes=[dtype, dtype],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-4,
        atol=1e-4,
        A=A,
        B=B,
        left=left,
    )


# solve_ex
@handle_frontend_test(
    fn_tree="torch.linalg.solve_ex",
    A=helpers.get_first_solve_batch_matrix(),
    B=helpers.get_second_solve_batch_matrix(choose_side=True),
    check_errors=st.booleans(),
)
def test_torch_solve_ex(
    *,
    A,
    B,
    check_errors,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, A, _ = A
    _, B, left = B
    helpers.test_frontend_function(
        input_dtypes=[dtype, dtype],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-4,
        atol=1e-4,
        A=A,
        B=B,
        left=left,
        check_errors=check_errors,
    )


# svd
@handle_frontend_test(
    fn_tree="torch.linalg.svd",
    dtype_and_x=_get_dtype_and_matrix(square=True),
    full_matrices=st.booleans(),
)
def test_torch_svd(
    *,
    dtype_and_x,
    full_matrices,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        atol=1e-03,
        rtol=1e-05,
        A=x,
        full_matrices=full_matrices,
    )
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
        backend=backend_fw,
    )


# svdvals
@handle_frontend_test(
    fn_tree="torch.linalg.svdvals",
    dtype_and_x=_get_dtype_and_matrix(batch=True),
    driver=st.sampled_from([None, "gesvd", "gesvdj", "gesvda"]),
)
def test_torch_svdvals(
    *,
    dtype_and_x,
    driver,
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
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        driver=driver,
        A=x[0],
    )


@handle_frontend_test(
    fn_tree="torch.linalg.tensorinv", dtype_input_ind=_tensorinv_helper()
)
def test_torch_tensorinv(
    *,
    dtype_input_ind,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, ind = dtype_input_ind
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-04,
        atol=1e-03,
        input=x,
        ind=ind,
    )


@handle_frontend_test(
    fn_tree="torch.linalg.tensorsolve",
    a_and_b=_get_solve_matrices(),
)
def test_torch_tensorsolve(
    *,
    a_and_b,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, A, B = a_and_b
    test_flags.num_positional_args = len(a_and_b) - 1
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-3,
        rtol=1e-3,
        A=A,
        B=B,
    )


@handle_frontend_test(
    fn_tree="torch.linalg.vander",
    dtype_and_input=_vander_helper(),
)
def test_torch_vander(
    *,
    dtype_and_input,
    frontend,
    fn_tree,
    on_device,
    test_flags,
    backend_fw,
):
    input_dtype, x, N = dtype_and_input
    test_flags.num_positional_args = 1
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_flags=test_flags,
        x=x[0],
        N=N,
    )


# vecdot
@handle_frontend_test(
    fn_tree="torch.linalg.vecdot",
    dtype_input_other_dim=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-1e3,
        max_value=1e3,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_torch_vecdot(
    dtype_input_other_dim,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    dtype, input, other, dim = dtype_input_other_dim
    test_flags.num_positional_args = len(dtype_input_other_dim) - 2
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        rtol=1e-2,
        atol=1e-3,
        x=input,
        y=other,
        dim=dim,
    )


# vector_norm
@handle_frontend_test(
    fn_tree="torch.linalg.vector_norm",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        valid_axis=True,
        abs_smallest_val=1e04,
    ),
    kd=st.booleans(),
    ord=st.one_of(
        helpers.ints(min_value=0, max_value=5),
        helpers.floats(min_value=1.0, max_value=5.0),
        st.sampled_from((float("inf"), -float("inf"))),
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_vector_norm(
    *,
    dtype_values_axis,
    kd,
    ord,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        ord=ord,
        dim=axis,
        keepdim=kd,
        dtype=dtype[0],
    )
