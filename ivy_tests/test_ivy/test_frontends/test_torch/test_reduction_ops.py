# global
import numpy as np
from hypothesis import given, strategies as st
import pytest

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


@pytest.mark.filterwarnings("ignore: dist")
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="dist"),
    dtype_and_y=statistical_dtype_values(function="dist"),
    as_variable=helpers.array_bools(num_arrays=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.dist"
    ),
    native_array=helpers.array_bools(num_arrays=2),
    p=st.integers(),
)
def test_torch_dist(
    dtype_and_x,
    dtype_and_y,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    p,
):
    input_x_dtype, x, dim_x = dtype_and_x
    input_y_dtype, y, dim_y = dtype_and_y
    if is_broadcastable(np.array(x).shape, np.array(y).shape):
        helpers.test_frontend_function(
            input_dtypes=[input_x_dtype],
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="torch",
            fn_tree="dist",
            input=np.asarray(x, dtype=input_x_dtype),
            other=np.asarray(y, dtype=input_y_dtype),
            p=p,
        )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.argmax"
    ),
    keepdims=st.booleans(),
)
def test_torch_argmax(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
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
        dim=axis,
        keepdim=keepdims,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.argmin"
    ),
    keepdims=st.booleans(),
)
def test_torch_argmin(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
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
        dim=axis,
        keepdim=keepdims,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.amax"
    ),
    keepdims=st.booleans(),
)
def test_torch_amax(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
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
        dim=axis,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.amin"
    ),
    keepdims=st.booleans(),
)
def test_torch_amin(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
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
        dim=axis,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        allow_inf=False,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.all"
    ),
    keepdims=st.booleans(),
)
def test_torch_all(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="all",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        allow_inf=False,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.any"
    ),
    keepdims=st.booleans(),
)
def test_torch_any(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="any",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        keepdim=keepdims,
        out=None,
    )
