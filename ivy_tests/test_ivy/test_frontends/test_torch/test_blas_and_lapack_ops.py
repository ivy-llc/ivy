# global
import sys
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


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
def _get_dtype_input_and_vectors(draw, with_input=False, same_size=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = dim_size1 if same_size else draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
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


@st.composite
def _get_dtype_input_and_matrices(draw, with_input=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
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
def _get_dtype_and_3dbatch_matrices(draw, with_input=False, input_3d=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
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
def _get_dtype_input_and_mat_vec(draw, *, with_input=False, skip_float16=False):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=True))

    if skip_float16:
        dtype = tuple(set(dtype).difference({"float16"}))

    dtype = draw(st.sampled_from(dtype))

    mat = draw(
        helpers.array_values(
            dtype=dtype, shape=(dim_size, shared_size), min_value=2, max_value=5
        )
    )
    vec = draw(
        helpers.array_values(
            dtype=dtype, shape=(shared_size,), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype, shape=(dim_size,), min_value=2, max_value=5
            )
        )
        return [dtype], input, mat, vec
    return [dtype], mat, vec


# addbmm
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.addbmm"
    ),
)
def test_torch_addbmm(
    dtype_and_matrices,
    beta,
    alpha,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, input, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="addbmm",
        rtol=1e-01,
        input=input,
        batch1=mat1,
        batch2=mat2,
        beta=beta,
        alpha=alpha,
        out=None,
    )


# addmm
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.addmm"
    ),
)
def test_torch_addmm(
    dtype_and_matrices,
    beta,
    alpha,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, input, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="addmm",
        rtol=1e-01,
        input=input,
        mat1=mat1,
        mat2=mat2,
        beta=beta,
        alpha=alpha,
        out=None,
    )


# addmv
@handle_cmd_line_args
@given(
    dtype_and_matrices=_get_dtype_input_and_mat_vec(with_input=True, skip_float16=True),
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.addmv"
    ),
)
def test_torch_addmv(
    dtype_and_matrices,
    beta,
    alpha,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, input, mat, vec = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="addmv",
        rtol=1e-03,
        input=input,
        mat=mat,
        vec=vec,
        beta=beta,
        alpha=alpha,
        out=None,
    )


# addr
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.addr"
    ),
)
def test_torch_addr(
    dtype_and_vecs,
    beta,
    alpha,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, input, vec1, vec2 = dtype_and_vecs

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="addr",
        rtol=1e-01,
        input=input,
        vec1=vec1,
        vec2=vec2,
        beta=beta,
        alpha=alpha,
        out=None,
    )


# baddbmm
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.baddbmm"
    ),
)
def test_torch_baddbmm(
    dtype_and_matrices,
    beta,
    alpha,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, input, batch1, batch2 = dtype_and_matrices

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="baddbmm",
        rtol=1e-01,
        input=input,
        batch1=batch1,
        batch2=batch2,
        beta=beta,
        alpha=alpha,
        out=None,
    )


# bmm
@handle_cmd_line_args
@given(
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.bmm"
    ),
)
def test_torch_bmm(
    dtype_and_matrices,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="bmm",
        rtol=1e-02,
        input=mat1,
        mat2=mat2,
        out=None,
    )


# cholesky
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1])) != 0
    ),
    upper=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.cholesky"
    ),
)
def test_torch_cholesky(
    dtype_and_x,
    upper,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, x = dtype_and_x
    x = x[0]
    x = (
        np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    )  # make symmetric positive-definite
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="cholesky",
        rtol=1e-02,
        input=x,
        upper=upper,
    )


# ger
@handle_cmd_line_args
@given(
    dtype_and_vecs=_get_dtype_input_and_vectors(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.ger"
    ),
)
def test_torch_ger(
    dtype_and_vecs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="ger",
        input=vec1,
        vec2=vec2,
    )


# inverse
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        min_value=0,
        max_value=25,
        shape=helpers.ints(min_value=2, max_value=10).map(lambda x: tuple([x, x])),
    ).filter(lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.inverse"
    ),
)
def test_torch_inverse(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="inverse",
        rtol=1e-03,
        input=x[0],
    )


# det
@handle_cmd_line_args
@given(
    dtype_and_x=_get_dtype_and_square_matrix(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.det"
    ),
)
def test_torch_det(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="det",
        input=x,
    )


# logdet
@handle_cmd_line_args
@given(
    dtype_and_x=_get_dtype_and_square_matrix(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.logdet"
    ),
)
def test_torch_logdet(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="logdet",
        input=x,
    )


# slogdet
@handle_cmd_line_args
@given(
    dtype_and_x=_get_dtype_and_square_matrix(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.slogdet"
    ),
)
def test_torch_slogdet(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="slogdet",
        input=x,
    )


# matmul
@handle_cmd_line_args
@given(
    dtype_xy=_get_dtype_and_3dbatch_matrices(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.matmul"
    ),
)
def test_torch_matmul(
    dtype_xy,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x, y = dtype_xy
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="matmul",
        rtol=1e-02,
        input=x,
        other=y,
        out=None,
    )


# matrix_power
@handle_cmd_line_args
@given(
    dtype_and_x=_get_dtype_and_square_matrix(),
    n=helpers.ints(min_value=2, max_value=5),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.matrix_power"
    ),
)
def test_torch_matrix_power(
    dtype_and_x,
    n,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="matrix_power",
        rtol=1e-01,
        input=x,
        n=n,
        out=None,
    )


# matrix_rank
@handle_cmd_line_args
@given(
    dtype_and_x=_get_dtype_and_square_matrix(),
    rtol=st.floats(1e-05, 1e-03),
    sym=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.matrix_rank"
    ),
)
def test_torch_matrix_rank(
    dtype_and_x,
    rtol,
    sym,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="matrix_rank",
        rtol=1e-01,
        input=x,
        tol=rtol,
        symmetric=sym,
    )


# mm
@handle_cmd_line_args
@given(
    dtype_xy=_get_dtype_input_and_matrices(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.mm"
    ),
)
def test_torch_mm(
    dtype_xy,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x, y = dtype_xy
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="mm",
        rtol=1e-03,
        input=x,
        mat2=y,
        out=None,
    )


# mv
@handle_cmd_line_args
@given(
    dtype_mat_vec=_get_dtype_input_and_mat_vec(skip_float16=True),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.mv"
    ),
)
def test_torch_mv(
    dtype_mat_vec,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, mat, vec = dtype_mat_vec
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="mv",
        rtol=1e-03,
        input=mat,
        vec=vec,
        out=None,
    )


# outer
@handle_cmd_line_args
@given(
    dtype_and_vecs=_get_dtype_input_and_vectors(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.outer"
    ),
)
def test_torch_outer(
    dtype_and_vecs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="outer",
        input=vec1,
        vec2=vec2,
        out=None,
    )


# pinverse
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.pinverse"
    ),
    rtol=st.floats(1e-5, 1e-3),
)
def test_torch_pinverse(
    dtype_and_x,
    rtol,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="pinverse",
        rtol=1e-03,
        input=x[0],
        rcond=rtol,
    )


# qr
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=2,
        max_value=5,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.qr"
    ),
    some=st.booleans(),
)
def test_torch_qr(
    dtype_and_x,
    some,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="qr",
        rtol=1e-02,
        input=x[0],
        some=some,
        out=None,
    )


# svd
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.svd"
    ),
    some=st.booleans(),
    compute=st.booleans(),
)
def test_torch_svd(
    dtype_and_x,
    some,
    compute,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="svd",
        input=x[0],
        some=some,
        compute_uv=compute,
        out=None,
    )


# vdot
@handle_cmd_line_args
@given(
    dtype_and_vecs=_get_dtype_input_and_vectors(same_size=True),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.vdot"
    ),
)
def test_torch_vdot(
    dtype_and_vecs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="vdot",
        input=vec1,
        other=vec2,
    )
