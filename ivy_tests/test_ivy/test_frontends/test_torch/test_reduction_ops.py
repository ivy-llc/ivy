# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        allow_inf=False,
    ),
    as_variable=helpers.array_bools(num_arrays=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.dist"
    ),
    native_array=helpers.array_bools(num_arrays=2),
    p=st.sampled_from([None, st.integers()]),
)
def test_torch_dist(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    p,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="dist",
        input=np.asarray(input[0], dtype=input_dtype[0]),
        other=np.asarray(input[1], dtype=input_dtype[1]),
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
