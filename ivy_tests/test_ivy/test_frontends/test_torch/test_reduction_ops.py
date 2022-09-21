# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def statistical_dtype_values(draw, *, function):
    max_op = "linear"
    if function in ["mean", "prod", "std", "sum", "var"]:
        max_op = "log"
    dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("float"),
            large_abs_safety_factor=4,
            small_abs_safety_factor=2,
            safety_factor_scale=max_op,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            valid_axis=True,
            allow_neg_axes=False,
            min_axes_size=1,
        )
    )
    if function in ["std", "var"]:
        shape = np.asarray(values, dtype=dtype).shape
        size = np.asarray(values, dtype=dtype).size
        max_correction = np.min(shape)
        if size == 1:
            correction = 0
        elif isinstance(axis, int):
            correction = draw(
                helpers.ints(min_value=0, max_value=shape[axis] - 1)
                | helpers.floats(min_value=0, max_value=shape[axis] - 1)
            )
            return dtype, values, axis, correction
        else:
            correction = draw(
                helpers.ints(min_value=0, max_value=max_correction - 1)
                | helpers.floats(min_value=0, max_value=max_correction - 1)
            )
        return dtype, values, axis, correction
    return dtype, values, axis


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
    fw,
    keepdims,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="mean",
        input=np.asarray(x, dtype=input_dtype),
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
    fw,
    keepdims,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="std",
        input=np.asarray(x, dtype=input_dtype),
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
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="prod",
        input=np.asarray(x, dtype=input_dtype),
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
    fw,
    keepdims,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="var",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        unbiased=bool(correction),
        keepdim=keepdims,
        out=None,
    )
