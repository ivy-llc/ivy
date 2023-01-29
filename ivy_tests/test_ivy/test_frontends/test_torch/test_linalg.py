# global
import sys
import numpy as np
from hypothesis import strategies as st


# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_torch.test_miscellaneous_ops import (
    dtype_value1_value2_axis,
)


# helpers
@st.composite
def _get_dtype_and_square_matrix(draw):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    mat = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=0, max_value=10
        )
    )
    return dtype, mat


@st.composite
def _get_dtype_and_matrix(draw):
    arbitrary_dims = draw(helpers.get_shape(max_dim_size=5))
    random_size = draw(st.integers(min_value=1, max_value=4))
    shape = (*arbitrary_dims, random_size, random_size)
    return draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape,
            min_value=-10,
            max_value=10,
        )
    )


# vector_norm
@handle_frontend_test(
    fn_tree="torch.linalg.vector_norm",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        valid_axis=True,
        min_value=-1e04,
        max_value=1e04,
    ),
    kd=st.booleans(),
    ord=helpers.ints(min_value=1, max_value=2),
    dtype=helpers.get_dtypes("valid"),
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
):
    dtype, x, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
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


# inv
@handle_frontend_test(
    fn_tree="torch.linalg.inv",
    aliases=["torch.inverse"],
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e4,
        max_value=1e4,
        small_abs_safety_factor=3,
        shape=helpers.ints(min_value=2, max_value=10).map(lambda x: tuple([x, x])),
    ).filter(lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon),
)
def test_torch_inv(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        atol=1e-02,
        input=x[0],
    )


# inv_ex
@handle_frontend_test(
    fn_tree="torch.linalg.inv_ex",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        min_value=0,
        max_value=20,
        shape=helpers.ints(min_value=2, max_value=10).map(lambda x: tuple([x, x])),
    ).filter(lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon),
)
def test_torch_inv_ex(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        atol=1e-02,
        input=x[0],
    )


# pinv
@handle_frontend_test(
    fn_tree="torch.linalg.pinv",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        min_value=-1e4,
        max_value=1e4,
    ),
    test_with_out=st.just(False),
)
def test_torch_pinv(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        atol=1e-02,
    )


# det
@handle_frontend_test(
    fn_tree="torch.linalg.det",
    aliases=["torch.det"],
    dtype_and_x=_get_dtype_and_square_matrix(),
)
def test_torch_det(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
    )


# qr
@handle_frontend_test(
    fn_tree="torch.linalg.qr",
    dtype_and_input=_get_dtype_and_matrix(),
    test_with_out=st.just(False),
)
def test_torch_qr(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
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
    )


# slogdet
@handle_frontend_test(
    fn_tree="torch.linalg.slogdet",
    aliases=["torch.slogdet"],
    dtype_and_x=_get_dtype_and_square_matrix(),
)
def test_torch_slogdet(
    *,
    dtype_and_x,
    fn_tree,
    frontend,
    on_device,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
    )


@st.composite
def _get_symmetrix_matrix(draw):
    input_dtype = draw(st.shared(st.sampled_from(draw(helpers.get_dtypes("float")))))
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


# eigvalsh
@handle_frontend_test(
    fn_tree="torch.linalg.eigvalsh",
    dtype_x=_get_symmetrix_matrix(),
    UPLO=st.sampled_from(("L", "U")),
    test_with_out=st.just(False),
)
def test_torch_eigvalsh(
    *,
    dtype_x,
    UPLO,
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
        input=x,
        UPLO=UPLO,
        atol=1e-4,
        rtol=1e-3,
    )


# matrix_power
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_power",
    aliases=["torch.matrix_power"],
    dtype_and_x=_get_dtype_and_square_matrix(),
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
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=x,
        n=n,
    )


# matrix_norm
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_norm",
    dtype_and_x=helpers.dtype_and_values(
        num_arrays=1,
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=5,
        min_value=-1e20,
        max_value=1e20,
        large_abs_safety_factor=10,
        small_abs_safety_factor=10,
        safety_factor_scale="log",
    ),
    ord=st.sampled_from(["fro", "nuc", np.inf, -np.inf, 1, -1, 2, -2]),
    keepdim=st.booleans(),
    axis=st.just((-2, -1)),
    dtype=helpers.get_dtypes("float", none=True, full=False),
)
def test_torch_matrix_norm(
    *,
    dtype_and_x,
    ord,
    keepdim,
    axis,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-04,
        atol=1e-04,
        input=x[0],
        ord=ord,
        dim=axis,
        keepdim=keepdim,
        dtype=dtype[0],
    )


# cross
@handle_frontend_test(
    fn_tree="torch.linalg.cross",
    dtype_input_other_dim=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("float"),
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
):
    dtype, input, other, dim = dtype_input_other_dim
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        rtol=1e-2,
        atol=1e-3,
        input=input,
        other=other,
        dim=dim,
    )


# vecdot
@handle_frontend_test(
    fn_tree="torch.linalg.vecdot",
    dtype_input_other_dim=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("float"),
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
):
    dtype, input, other, dim = dtype_input_other_dim
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        rtol=1e-2,
        atol=1e-3,
        input=input,
        other=other,
        dim=dim,
    )


@st.composite
def _matrix_rank_helper(draw):
    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=2,
            min_value=-1e05,
            max_value=1e05,
        )
    )
    return dtype_x


# matrix_rank
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_rank",
    aliases=["torch.matrix_rank"],
    dtype_and_x=_matrix_rank_helper(),
    atol=st.floats(min_value=1e-5, max_value=0.1, exclude_min=True, exclude_max=True),
    rtol=st.floats(min_value=1e-5, max_value=0.1, exclude_min=True, exclude_max=True),
)
def test_matrix_rank(
    dtype_and_x,
    rtol,
    atol,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        rtol=rtol,
        atol=atol,
    )


@handle_frontend_test(
    fn_tree="torch.linalg.cholesky",
    aliases=["torch.cholesky"],
    dtype_and_x=_get_dtype_and_square_matrix(),
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
):
    dtype, x = dtype_and_x
    x = np.matmul(x.T, x) + np.identity(x.shape[0])  # make symmetric positive-definite

    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=x,
        upper=upper,
    )


# svd
@handle_frontend_test(
    fn_tree="torch.linalg.svd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ),
    full_matrices=st.booleans(),
)
def test_torch_svd(
    *,
    dtype_and_x,
    full_matrices,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
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
    )


# eig
@handle_frontend_test(
    fn_tree="torch.linalg.eig",
    dtype_and_input=_get_dtype_and_square_matrix(),
    test_with_out=st.just(False),
)
def test_torch_eig(
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
        input=x,
    )
    ret = [ivy.to_numpy(x).astype("float64") for x in ret]
    frontend_ret = [np.asarray(x, dtype=np.float64) for x in frontend_ret]

    l, v = ret
    front_l, front_v = frontend_ret

    assert_all_close(
        ret_np=v @ np.diag(l) @ np.linalg.inv(v),
        ret_from_gt_np=front_v @ np.diag(front_l) @ np.linalg.inv(front_v),
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
    )


# svdvals
@handle_frontend_test(
    fn_tree="torch.linalg.svdvals",
    dtype_and_x=_get_dtype_and_square_matrix(),
)
def test_torch_svdvals(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        A=x,
    )


# solve
@handle_frontend_test(
    fn_tree="torch.linalg.solve",
    dtype_and_data=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x + 1])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0][:, :-1]) < 1 / sys.float_info.epsilon
        and np.linalg.det(x[1][0][:, :-1]) != 0
        and np.linalg.cond(x[1][0][:, -1].reshape(-1, 1)) < 1 / sys.float_info.epsilon
    ),
)
def test_torch_solve(
    *,
    dtype_and_data,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, data = dtype_and_data
    input = data[0][:, :-1]
    other = data[0][:, -1].reshape(-1, 1)
    helpers.test_frontend_function(
        input_dtypes=[input_dtype[0], input_dtype[0]],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        other=other,
    )


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
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape_draw,
        ).filter(lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon)
    )
    input[0] = input[0].reshape(shape_cor)
    return dtype, input[0], ind


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
):

    dtype, x, ind = dtype_input_ind
    helpers.test_frontend_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-04,
        atol=1e-03,
        input=x,
        ind=ind,
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
):
    input_dtype, A, B = a_and_b
    test_flags.num_positional_args = 2
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-3,
        rtol=1e-3,
        A=A,
        B=B,
    )
