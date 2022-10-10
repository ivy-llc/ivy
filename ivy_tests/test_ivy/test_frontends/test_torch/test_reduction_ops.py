# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)


@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        allow_inf=False,
    ),
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
    p,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="dist",
        input=input[0],
        other=input[1],
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
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="argmax",
        input=x[0],
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
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="argmin",
        input=x[0],
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
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="amax",
        input=x[0],
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
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="amin",
        input=x[0],
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
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="all",
        input=x[0],
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
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="any",
        input=x[0],
        dim=axis,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="mean"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.mean"
    ),
    keepdims=st.booleans(),
)
def test_torch_mean(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    keepdims,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="mean",
        input=x[0],
        dim=axis,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="std"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.std"
    ),
    keepdims=st.booleans(),
)
def test_torch_std(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    keepdims,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="std",
        input=x[0],
        dim=axis,
        unbiased=bool(correction),
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("numeric", none=True),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.prod"
    ),
    keepdims=st.booleans(),
)
def test_torch_prod(
    dtype_x_axis,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    keepdims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="prod",
        x=x[0],
        dim=axis,
        dtype=dtype,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="var"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.var"
    ),
    keepdims=st.booleans(),
)
def test_torch_var(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    keepdims,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="var",
        input=x[0],
        dim=axis,
        unbiased=bool(correction),
        keepdim=keepdims,
        out=None,
    )


# ToDo, fails for TensorFlow backend, tf.reduce_min doesn't support bool
# ToDo, fails for torch backend, tf.argmin_cpu doesn't support bool
@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.min"
    ),
    keepdim=st.booleans(),
)
def test_torch_min(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdim,
    with_out,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="min",
        input=x[0],
        dim=axis,
        keepdim=keepdim,
        out=None,
    )
