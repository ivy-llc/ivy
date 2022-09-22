# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# allclose
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.allclose"
    ),
    equal_nan=st.booleans(),
)
def test_torch_allclose(
    dtype_and_input,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="allclose",
        input=np.asarray(input[0], dtype=input_dtype[0]),
        other=np.asarray(input[1], dtype=input_dtype[1]),
        rtol=1e-05,
        atol=1e-08,
        equal_nan=equal_nan,
    )


# equal
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.equal"
    ),
)
def test_torch_equal(
    dtype_and_inputs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    inputs_dtypes, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=inputs_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="equal",
        input=np.asarray(inputs[0], dtype=inputs_dtypes[0]),
        other=np.asarray(inputs[1], dtype=inputs_dtypes[1]),
    )


# eq
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.eq"
    ),
)
def test_torch_eq(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    inputs_dtypes, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=inputs_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="eq",
        input=np.asarray(inputs[0], dtype=inputs_dtypes[0]),
        other=np.asarray(inputs[1], dtype=inputs_dtypes[1]),
        out=None,
    )


# argsort
@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.argsort"
    ),
    descending=st.booleans(),
)
def test_torch_argsort(
    dtype_input_axis,
    descending,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="argsort",
        input=np.asarray(input, dtype=input_dtype),
        dim=axis,
        descending=descending,
    )


# greater_equal
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.greater_equal"
    ),
)
def test_torch_greater_equal(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="greater_equal",
        input=np.asarray(inputs[0], dtype=input_dtype[0]),
        other=np.asarray(inputs[1], dtype=input_dtype[1]),
        out=None,
    )


# greater
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.greater"
    ),
)
def test_torch_greater(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="greater",
        input=np.asarray(inputs[0], dtype=input_dtype[0]),
        other=np.asarray(inputs[1], dtype=input_dtype[1]),
        out=None,
    )


# isclose
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isclose"
    ),
    equal_nan=st.booleans(),
)
def test_torch_isclose(
    dtype_and_input,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="isclose",
        input=np.asarray(input[0], dtype=input_dtype[0]),
        other=np.asarray(input[1], dtype=input_dtype[1]),
        rtol=1e-05,
        atol=1e-08,
        equal_nan=equal_nan,
    )


# isifinite
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isfinite"
    ),
)
def test_torch_isfinite(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="isfinite",
        input=np.asarray(input, dtype=input_dtype),
    )


# isinf
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isinf"
    ),
)
def test_torch_isinf(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="isinf",
        input=np.asarray(input, dtype=input_dtype),
    )


# isposinf
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isposinf"
    ),
)
def test_torch_isposinf(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="isposinf",
        input=np.asarray(input, dtype=input_dtype),
    )


# isneginf
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isneginf"
    ),
)
def test_torch_isneginf(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="isneginf",
        input=np.asarray(input, dtype=input_dtype),
    )


# sort
@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        min_axis=-1,
        max_axis=0,
    ),
    descending=st.booleans(),
    stable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.sort"
    ),
)
def test_torch_sort(
    dtype_input_axis,
    descending,
    stable,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="sort",
        input=np.asarray(input, dtype=input_dtype),
        dim=axis,
        descending=descending,
        stable=stable,
        out=None,
    )


# isnan
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isnan"
    ),
)
def test_torch_isnan(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="isnan",
        input=np.asarray(input, dtype=input_dtype),
    )


# less_equal
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.less_equal"
    ),
)
def test_torch_less_equal(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="less_equal",
        input=np.asarray(inputs[0], dtype=input_dtype[0]),
        other=np.asarray(inputs[1], dtype=input_dtype[1]),
        out=None,
    )


# less
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.less"
    ),
)
def test_torch_less(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="less",
        input=np.asarray(inputs[0], dtype=input_dtype[0]),
        other=np.asarray(inputs[1], dtype=input_dtype[1]),
        out=None,
    )


# not_equal
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.not_equal"
    ),
)
def test_torch_not_equal(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="not_equal",
        input=np.asarray(inputs[0], dtype=input_dtype[0]),
        other=np.asarray(inputs[1], dtype=input_dtype[1]),
        out=None,
    )


# minimum
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.minimum"
    ),
)
def test_torch_minimum(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="minimum",
        input=np.asarray(inputs[0], dtype=input_dtype[0]),
        other=np.asarray(inputs[1], dtype=input_dtype[1]),
        out=None,
    )


# fmax
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=True,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.fmax"
    ),
)
def test_torch_fmax(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="fmax",
        input=np.asarray(inputs[0], dtype=input_dtype[0]),
        other=np.asarray(inputs[1], dtype=input_dtype[1]),
        out=None,
    )


# fmin
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=True,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.fmin"
    ),
)
def test_torch_fmin(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="fmin",
        input=np.asarray(inputs[0], dtype=input_dtype[0]),
        other=np.asarray(inputs[1], dtype=input_dtype[1]),
        out=None,
    )


# msort
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.msort"
    ),
)
def test_torch_msort(
    dtype_and_input,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="msort",
        input=np.asarray(input, dtype=input_dtype),
    )
