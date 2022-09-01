# global
from typing import Sequence
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.statistical_dtype_values(function="argmax"),
    as_variable=helpers.array_bools(num_arrays=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.argmax"
    ),
    native_array=helpers.array_bools(num_arrays=1),
    keepdims=st.booleans(),
)
def test_torch_argmax(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    fw,
):
    input_dtype, x, dim = dtype_and_x
    dim = dim[0] if isinstance(dim, Sequence) else dim
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="argmax",
        input=np.asarray(x, dtype=input_dtype),
        dim=dim,
        keepdim=keepdims,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.statistical_dtype_values(function="argmin"),
    as_variable=helpers.array_bools(num_arrays=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.argmin"
    ),
    native_array=helpers.array_bools(num_arrays=1),
    keepdims=st.booleans(),
)
def test_torch_argmin(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    fw,
):
    input_dtype, x, dim = dtype_and_x
    dim = dim[0] if isinstance(dim, Sequence) else dim
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="argmin",
        input=np.asarray(x, dtype=input_dtype),
        dim=dim,
        keepdim=keepdims,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.statistical_dtype_values(function="amax"),
    as_variable=helpers.array_bools(num_arrays=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.amax"
    ),
    native_array=helpers.array_bools(num_arrays=1),
    keepdims=st.booleans(),
    with_out=st.booleans(),
)
def test_torch_amax(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, dim = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="amax",
        input=np.asarray(x, dtype=input_dtype),
        dim=dim,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.statistical_dtype_values(function="amin"),
    as_variable=helpers.array_bools(num_arrays=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.amin"
    ),
    native_array=helpers.array_bools(num_arrays=1),
    keepdims=st.booleans(),
    with_out=st.booleans(),
)
def test_torch_amin(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, dim = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="amin",
        input=np.asarray(x, dtype=input_dtype),
        dim=dim,
        keepdim=keepdims,
        out=None,
    )
