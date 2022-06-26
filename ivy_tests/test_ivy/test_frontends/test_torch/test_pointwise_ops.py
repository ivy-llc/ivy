# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


# add
@given(
    dtype_and_x=helpers.dtype_and_values(
        tuple(set(ivy_np.valid_float_dtypes).intersection(
              set(ivy_torch.valid_float_dtypes))), 2),
    alpha=st.floats(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.add"),
    native_array=st.booleans(),
)
def test_torch_add(
    dtype_and_x,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
        "torch",
        "add",
        input=np.asarray(x[0], dtype=input_dtype[0]),
        other=np.asarray(x[1], dtype=input_dtype[1]),
        alpha=alpha,
        out=None,
    )


# tan
@given(
    dtype_and_x=helpers.dtype_and_values(
        tuple(set(ivy_np.valid_float_dtypes).intersection(
              set(ivy_torch.valid_float_dtypes)))),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.tan"),
    native_array=st.booleans(),
)
def test_torch_tan(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
        "torch",
        "tan",
        input=np.asarray(x, dtype=input_dtype),
        out=None,
    )
