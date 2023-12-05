# global
import sys
import numpy as np
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import _matrix_rank_helper
from ivy_tests.test_ivy.helpers.hypothesis_helpers.general_helpers import (
    matrix_is_stable,
)


# --- Helpers --- #
# --------------- #


@st.composite
def _generate_chain_matmul_dtype_and_arrays(draw):
    dtype = draw(helpers.get_dtypes("float", full=True))
    input_dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]
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
def _get_dtype_and_3dbatch_matrices(draw, with_input=False, input_3d=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", full=True))
    dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]
    batch_size = draw(helpers.ints(min_value=2, max_value=4))
    mat1 = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(batch_size, dim_size1, shared_size),
            min_value=2,
            max_value=5,
        )
    )
    mat2 = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(batch_size, shared_size, dim_size2),
            min_value=2,
            max_value=5,
        )
    )
    if with_input:
        if input_3d:
            input = draw(
                helpers.array_values(
                    dtype=dtype[0],
                    shape=(batch_size, dim_size1, dim_size2),
                    min_value=2,
                    max_value=5,
                )
            )
            return dtype, input, mat1, mat2
        input = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size1, dim_size2), min_value=2, max_value=5
            )
        )
        return dtype, input, mat1, mat2
    return dtype, mat1, mat2


@st.composite
def _get_dtype_and_matrices(draw):
    dim1 = draw(helpers.ints(min_value=2, max_value=7))
    dim2 = draw(helpers.ints(min_value=2, max_value=7))
    dtype = draw(helpers.get_dtypes("float", full=False))

    matr1 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim1, dim2), min_value=2, max_value=10
        )
    )
    matr2 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim1, dim2), min_value=2, max_value=10
        )
    )

    return dtype, matr1, matr2


# helpers
@st.composite
def _get_dtype_and_square_matrix(draw):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", full=True))
    dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]
    mat = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=0, max_value=10
        )
    )
    return dtype, mat


@st.composite
def _get_dtype_input_and_mat_vec(draw, *, with_input=False):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", full=True))
    dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]

    mat = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, shared_size), min_value=2, max_value=5
        )
    )
    vec = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(shared_size,), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
        return dtype, input, mat, vec
    return dtype, mat, vec


@st.composite
def _get_dtype_input_and_matrices(draw, with_input=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", full=True))
    dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]
    mat1 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size1, shared_size), min_value=2, max_value=5
        )
    )
    mat2 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(shared_size, dim_size2), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size1, dim_size2), min_value=2, max_value=5
            )
        )
        return dtype, input, mat1, mat2
    return dtype, mat1, mat2


@st.composite
def _get_dtype_input_and_vectors(draw, with_input=False, same_size=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = dim_size1 if same_size else draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", full=True))
    dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]
    vec1 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size1,), min_value=2, max_value=5
        )
    )
    vec2 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size2,), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size1, dim_size2), min_value=2, max_value=5
            )
        )
        return dtype, input, vec1, vec2
    return dtype, vec1, vec2


# --- Main --- #
# ------------ #


# addbmm
@handle_frontend_test(
    fn_tree="torch.addbmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addbmm(
    dtype_and_matrices,
    beta,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=input,
        batch1=mat1,
        batch2=mat2,
        beta=beta,
        alpha=alpha,
    )


# addmm
@handle_frontend_test(
    fn_tree="torch.addmm",
    dtype_and_matrices=_get_dtype_input_and_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addmm(
    dtype_and_matrices,
    beta,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=input,
        mat1=mat1,
        mat2=mat2,
        beta=beta,
        alpha=alpha,
    )


# addmv
@handle_frontend_test(
    fn_tree="torch.addmv",
    dtype_and_matrices=_get_dtype_input_and_mat_vec(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addmv(
    dtype_and_matrices,
    beta,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input, mat, vec = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        input=input,
        mat=mat,
        vec=vec,
        beta=beta,
        alpha=alpha,
    )


# addr
@handle_frontend_test(
    fn_tree="torch.addr",
    dtype_and_vecs=_get_dtype_input_and_vectors(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addr(
    dtype_and_vecs,
    beta,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input, vec1, vec2 = dtype_and_vecs

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=input,
        vec1=vec1,
        vec2=vec2,
        beta=beta,
        alpha=alpha,
    )


# baddbmm
@handle_frontend_test(
    fn_tree="torch.baddbmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True, input_3d=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_baddbmm(
    dtype_and_matrices,
    beta,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input, batch1, batch2 = dtype_and_matrices

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=input,
        batch1=batch1,
        batch2=batch2,
        beta=beta,
        alpha=alpha,
    )


# bmm
@handle_frontend_test(
    fn_tree="torch.bmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(),
)
def test_torch_bmm(
    dtype_and_matrices,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        input=mat1,
        mat2=mat2,
    )


# chain_matmul
@handle_frontend_test(
    fn_tree="torch.chain_matmul",
    dtype_and_matrices=_generate_chain_matmul_dtype_and_arrays(),
)
def test_torch_chain_matmul(
    *,
    dtype_and_matrices,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, matrices = dtype_and_matrices
    args = {f"x{i}": matrix for i, matrix in enumerate(matrices)}
    test_flags.num_positional_args = len(matrices)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        **args,
    )


# cholesky
@handle_frontend_test(
    fn_tree="torch.cholesky",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: (x, x)),
    ).filter(
        lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1])) != 0
    ),
    upper=st.booleans(),
)
def test_torch_cholesky(
    dtype_and_x,
    upper,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    x = x[0]
    x = (
        np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    )  # make symmetric positive-definite
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        input=x,
        upper=upper,
    )


# dot
@handle_frontend_test(
    fn_tree="torch.dot",
    dtype_and_vecs=_get_dtype_input_and_vectors(same_size=True),
)
def test_torch_dot(
    dtype_and_vecs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vec1, vec2 = dtype_and_vecs
    test_flags.num_positional_args = len(dtype_and_vecs) - 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=vec1,
        other=vec2,
    )


# ger
@handle_frontend_test(
    fn_tree="torch.ger",
    dtype_and_vecs=_get_dtype_input_and_vectors(),
)
def test_torch_ger(
    dtype_and_vecs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=vec1,
        vec2=vec2,
    )


# inner
@handle_frontend_test(
    fn_tree="torch.inner",
    dtype_and_matrices=_get_dtype_input_and_matrices(with_input=True),
)
def test_torch_inner(
    dtype_and_matrices,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input_mat, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input_mat,
        other=mat2,
        out=None,
    )


# logdet
@handle_frontend_test(
    fn_tree="torch.logdet",
    dtype_and_x=_get_dtype_and_square_matrix(),
)
def test_torch_logdet(
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
        input=x,
    )


# matmul
@handle_frontend_test(
    fn_tree="torch.matmul",
    dtype_xy=_get_dtype_and_3dbatch_matrices(),
)
def test_torch_matmul(
    dtype_xy,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, y = dtype_xy
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        input=x,
        other=y,
        out=None,
    )


# matrix_rank
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_rank",
    # aliases=["torch.matrix_rank",], deprecated since 1.9. uncomment with multi-version
    # testing pipeline
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
        A=x,
        atol=atol,
        rtol=rtol,
        hermitian=hermitian,
    )


# mm
@handle_frontend_test(
    fn_tree="torch.mm",
    dtype_xy=_get_dtype_input_and_matrices(),
)
def test_torch_mm(
    dtype_xy,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, y = dtype_xy
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-02,
        rtol=1e-02,
        input=x,
        mat2=y,
    )


# mv
@handle_frontend_test(
    fn_tree="torch.mv",
    dtype_mat_vec=_get_dtype_input_and_mat_vec(),
)
def test_torch_mv(
    dtype_mat_vec,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, mat, vec = dtype_mat_vec
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        input=mat,
        vec=vec,
        out=None,
    )


# outer
@handle_frontend_test(
    fn_tree="torch.outer",
    dtype_and_vecs=_get_dtype_input_and_vectors(),
)
def test_torch_outer(
    dtype_and_vecs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=vec1,
        vec2=vec2,
    )


# pinverse
@handle_frontend_test(
    fn_tree="torch.pinverse",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
    ),
    rtol=st.floats(1e-5, 1e-3),
)
def test_torch_pinverse(
    dtype_and_x,
    rtol,
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
        input=x[0],
        rcond=rtol,
    )


# qr
@handle_frontend_test(
    fn_tree="torch.qr",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=2,
        max_value=5,
    ),
    some=st.booleans(),
)
def test_torch_qr(
    dtype_and_x,
    some,
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
        rtol=1e-02,
        input=x[0],
        some=some,
    )


# svd
@handle_frontend_test(
    fn_tree="torch.svd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    some=st.booleans(),
    compute=st.booleans(),
)
def test_torch_svd(
    dtype_and_x,
    some,
    compute,
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
        input=x[0],
        some=some,
        compute_uv=compute,
    )


@handle_frontend_test(
    fn_tree="torch.trapezoid",
    test_with_out=st.just(False),
    dtype_y_x=_get_dtype_and_matrices(),
    use_x=st.booleans(),
    dim=st.integers(min_value=0, max_value=1),
    dx=st.floats(),
)
def test_torch_trapezoid(
    dtype_y_x,
    use_x,
    dim,
    dx,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, y, x = dtype_y_x
    if use_x:
        test_flags.num_positional_args = 2
        kwargs = {"y": y, "x": x, "dim": -1}
    else:
        test_flags.num_positional_args = 1
        kwargs = {"y": y, "dx": dx, "dim": dim}
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **kwargs,
    )


# vdot
@handle_frontend_test(
    fn_tree="torch.vdot",
    dtype_and_vecs=_get_dtype_input_and_vectors(same_size=True),
)
def test_torch_vdot(
    dtype_and_vecs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=vec1,
        other=vec2,
    )
