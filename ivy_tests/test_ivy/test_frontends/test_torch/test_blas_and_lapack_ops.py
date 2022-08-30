# global
import sys
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# helpers
@st.composite
def _get_dtype_and_batch_matrices(draw):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1))
    shape1 = (dim_size1, shared_size)
    shape2 = (shared_size, dim_size2)
    batched = draw(st.booleans())
    if batched:
        batch_size1 = draw(helpers.ints(min_value=2, max_value=4))
        batch_size2 = draw(helpers.ints(min_value=2, max_value=4))
        shape1 = (batch_size1, 1, dim_size1, shared_size)
        shape2 = (batch_size2, shared_size, dim_size2)
    mat1 = draw(
        helpers.array_values(dtype=dtype, shape=shape1, min_value=2, max_value=5)
    )
    mat2 = draw(
        helpers.array_values(dtype=dtype, shape=shape2, min_value=2, max_value=5)
    )
    return dtype, mat1, mat2


@st.composite
def _get_dtype_and_square_matrix(draw):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1))
    mat = draw(
        helpers.array_values(
            dtype=dtype, shape=(dim_size, dim_size), min_value=0, max_value=10
        )
    )
    return dtype, mat


@st.composite
def _get_dtype_input_and_matrices(draw, with_input=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1))
    mat1 = draw(
        helpers.array_values(
            dtype=dtype, shape=(dim_size1, shared_size), min_value=2, max_value=5
        )
    )
    mat2 = draw(
        helpers.array_values(
            dtype=dtype, shape=(shared_size, dim_size2), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype, shape=(dim_size1, dim_size2), min_value=2, max_value=5
            )
        )
        return dtype, input, mat1, mat2
    return dtype, mat1, mat2


@st.composite
def _get_dtype_and_3dbatch_matrices(draw):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1))
    batch_size = draw(helpers.ints(min_value=2, max_value=4))
    mat1 = draw(
        helpers.array_values(
            dtype=dtype,
            shape=(batch_size, dim_size1, shared_size),
            min_value=2,
            max_value=5,
        )
    )
    mat2 = draw(
        helpers.array_values(
            dtype=dtype,
            shape=(batch_size, shared_size, dim_size2),
            min_value=2,
            max_value=5,
        )
    )
    return dtype, mat1, mat2


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
        input_dtypes=[dtype, dtype, dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="addmm",
        rtol=1e-01,
        input=np.asarray(input, dtype=dtype),
        mat1=np.asarray(mat1, dtype=dtype),
        mat2=np.asarray(mat2, dtype=dtype),
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
        input_dtypes=[dtype, dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="bmm",
        rtol=1e-02,
        input=np.asarray(mat1, dtype=dtype),
        mat2=np.asarray(mat2, dtype=dtype),
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
    x = np.asarray(x, dtype=dtype)
    x = (
        np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    )  # make symmetric positive-definite

    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="cholesky",
        rtol=1e-02,
        input=np.asarray(x, dtype=dtype),
        upper=upper,
    )


# ger
@handle_cmd_line_args
@given(
    dtype_xy=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        num_arrays=2,
        min_value=1,
        max_value=50,
        min_num_dims=1,
        max_num_dims=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.ger"
    ),
)
def test_torch_ger(
    dtype_xy,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    types, arrays = dtype_xy
    type1, type2 = types
    x1, x2 = arrays

    helpers.test_frontend_function(
        input_dtypes=types,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="ger",
        input=np.asarray(x1, dtype=type1),
        vec2=np.asarray(x2, dtype=type2),
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
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="inverse",
        rtol=1e-03,
        input=np.asarray(x, dtype=dtype),
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
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="det",
        input=np.asarray(x, dtype=dtype),
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
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="slogdet",
        input=np.asarray(x, dtype=dtype),
    )


# matmul
@handle_cmd_line_args
@given(
    dtype_xy=_get_dtype_and_batch_matrices(),
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
        input_dtypes=[dtype, dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="matmul",
        rtol=1e-03,
        input=np.asarray(x, dtype=dtype),
        other=np.asarray(y, dtype=dtype),
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
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="matrix_power",
        rtol=1e-01,
        input=np.asarray(x, dtype=dtype),
        n=n,
        out=None,
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
        input_dtypes=[dtype, dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="mm",
        rtol=1e-03,
        input=np.asarray(x, dtype=dtype),
        mat2=np.asarray(y, dtype=dtype),
        out=None,
    )


# outer
@handle_cmd_line_args
@given(
    dtype_xy=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        num_arrays=2,
        min_value=1,
        max_value=50,
        min_num_dims=1,
        max_num_dims=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.outer"
    ),
)
def test_torch_outer(
    dtype_xy,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    types, arrays = dtype_xy
    type1, type2 = types
    x1, x2 = arrays

    helpers.test_frontend_function(
        input_dtypes=types,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="outer",
        input=np.asarray(x1, dtype=type1),
        vec2=np.asarray(x2, dtype=type2),
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
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="pinverse",
        rtol=1e-03,
        input=np.asarray(x, dtype=dtype),
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
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="qr",
        rtol=1e-02,
        input=np.array(x, dtype=dtype),
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
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="svd",
        input=np.array(x, dtype=dtype),
        some=some,
        compute_uv=compute,
        out=None,
    )
