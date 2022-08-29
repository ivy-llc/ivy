# global
import sys
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# cholesky
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(ivy_torch.valid_float_dtypes)
        ),
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
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                ivy_torch.valid_numeric_dtypes
            )
        ),
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
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(ivy_torch.valid_float_dtypes)
        ),
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


# outer
@handle_cmd_line_args
@given(
    dtype_xy=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                ivy_torch.valid_numeric_dtypes
            )
        ),
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


# qr
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(ivy_torch.valid_float_dtypes)
        ),
        min_num_dims=3,
        max_num_dims=5,
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
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(ivy_torch.valid_float_dtypes)
        ),
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
