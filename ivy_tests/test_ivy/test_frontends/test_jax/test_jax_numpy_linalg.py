# global
import sys
import numpy as np

from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close, handle_cmd_line_args
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
    _matrix_rank_helper,
)


# det
@handle_cmd_line_args
@given(
    dtype_and_x=_get_dtype_and_matrix(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.det"
    ),
)
def test_jax_numpy_det(dtype_and_x, as_variable, native_array, num_positional_args):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        fn_tree="numpy.linalg.det",
        rtol=1e-04,
        atol=1e-04,
        a=np.asarray(x[0], dtype=dtype[0]),
    )


# eigh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.eigh"
    ),
    UPLO=st.sampled_from(("L", "U")),
    symmetrize_input=st.booleans(),
)
def test_jax_numpy_eigh(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    UPLO,
    symmetrize_input,
):
    dtype, x = dtype_and_x
    x = np.array(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        fn_tree="numpy.linalg.eigh",
        test_values=False,
        a=x,
        UPLO=UPLO,
        symmetrize_input=symmetrize_input,
    )
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    L, Q = ret
    frontend_L, frontend_Q = frontend_ret

    assert_all_close(
        ret_np=Q @ np.diag(L) @ Q.T,
        ret_from_gt_np=frontend_Q @ np.diag(frontend_L) @ frontend_Q.T,
        atol=1e-02,
    )


# inv
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        shape=helpers.ints(min_value=1, max_value=10).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.inv"
    ),
)
def test_jax_numpy_inv(dtype_and_x, as_variable, native_array, num_positional_args):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        rtol=1e-01,
        atol=1e-01,
        frontend="jax",
        fn_tree="numpy.linalg.inv",
        a=np.asarray(x[0], dtype=dtype[0]),
    )


# eigvalsh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.eigvalsh"
    ),
    UPLO=st.sampled_from(("L", "U")),
)
def test_jax_numpy_eigvalsh(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    UPLO,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        fn_tree="numpy.linalg.eigvalsh",
        rtol=1e-02,
        atol=1e-02,
        a=x,
        UPLO=UPLO,
    )


# qr
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        min_value=2,
        max_value=5,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.qr"
    ),
    mode=st.sampled_from(("reduced", "complete")),
)
def test_jax_numpy_qr(
    dtype_and_x,
    mode,
    as_variable,
    native_array,
    num_positional_args,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        rtol=1e-01,
        atol=1e-01,
        frontend="jax",
        fn_tree="numpy.linalg.qr",
        a=np.asarray(x[0], dtype[0]),
        mode=mode,
    )


# eigvals
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.eigvals"
    ),
)
def test_jax_numpy_eigvals(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
):
    dtype, x = dtype_and_x
    x = np.array(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        fn_tree="numpy.linalg.eigvals",
        test_values=False,
        a=x,
    )


# cholesky
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(x[1][0]) != 0
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.cholesky"
    ),
)
def test_jax_numpy_cholesky(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    fw,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        fn_tree="numpy.linalg.cholesky",
        rtol=1e-02,
        a=x,
    )


# slogdet
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_value=100,
        min_value=-100,
        shape=st.tuples(
            st.shared(st.integers(1, 5), key="sq"),
            st.shared(st.integers(1, 5), key="sq"),
        ),
        num_arrays=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.slogdet"
    ),
)
def test_jax_slogdet(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        fn_tree="numpy.linalg.slogdet",
        a=np.asarray(x[0], dtype=input_dtype[0]),
    )


# matrix_rank
@handle_cmd_line_args
@given(
    dtype_and_x=_matrix_rank_helper(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.matrix_rank"
    ),
)
def test_jax_numpy_matrix_rank(
    dtype_and_x, as_variable, native_array, num_positional_args
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        fn_tree="numpy.linalg.matrix_rank",
        M=x[0],
    )


# solve
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.solve"
    ),
)
def test_jax_numpy_solve(
    dtype_and_data,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, data = dtype_and_data
    a = data[0][:, :-1]
    b = data[0][:, -1].reshape(-1, 1)
    helpers.test_frontend_function(
        input_dtypes=[input_dtype[0], input_dtype[0]],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        fn_tree="numpy.linalg.solve",
        a=np.asarray(a, dtype=input_dtype[0]),
        b=np.asarray(b, dtype=input_dtype[0]),
        rtol=1e-01,
        atol=1e-01,
    )


@st.composite
def norm_helper(draw):
    dtype, x, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=2,
            max_num_dims=4,
            min_dim_size=2,
            max_dim_size=4,
            min_axis=-1,
            max_axis=2,
            force_int_axis=False,
            max_axes_size=2,
            safety_factor_scale="log",
            valid_axis=True,
            large_abs_safety_factor=2,
        )
    )
    if type(axis) in [tuple, list]:
        if len(axis) == 2:
            ord_param = draw(
                st.sampled_from(["fro", "nuc", 1, 2, -1, -2, np.inf, -np.inf])
            )
        else:
            axis = axis[0]
            ord_param = draw(st.sampled_from([0, 1, 2, -1, -2, np.inf, -np.inf]))
    else:
        ord_param = draw(st.sampled_from([0, 1, 2, -1, -2, np.inf, -np.inf]))
    keepdims = draw(st.booleans())
    return dtype, x, ord_param, axis, keepdims


# norm
@handle_cmd_line_args
@given(
    params=norm_helper().filter(lambda s: "bfloat16" not in s[0] or "bool" not in s[0]),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.norm"
    ),
)
def test_jax_norm(
    params,
    as_variable,
    num_positional_args,
    native_array,
):
    dtype, x, ord_param, axis, keepdims = params

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        fn_tree="numpy.linalg.norm",
        x=x[0],
        ord=ord_param,
        axis=axis,
        keepdims=keepdims,
    )
