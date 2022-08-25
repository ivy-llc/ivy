# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# allclose
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ),
        ),
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
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
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
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
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
