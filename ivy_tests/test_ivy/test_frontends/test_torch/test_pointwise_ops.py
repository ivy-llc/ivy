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
        tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(min_value=-1e06, max_value=1e06, allow_infinity=False),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.add"
    ),
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
        rtol=1e-04,
    )


# tan
@given(
    dtype_and_x=helpers.dtype_and_values(
        tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        )
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.tan"
    ),
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

#cos
@given(
        dtype_and_x=helpers.dtype_and_values(
            tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_torch.valid_float_dtypes)
                )
            )
        ),
        as_variable = st.booleans(),
        with_out = st.booleans(),
        num_positional_args = helpers.num_positional_args(fn_name="functional.frontends.torch.cos"),
        native_array=st.booleans(),

)
def test_torch_cos(
        dtype_and_x,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
):
    input_dtype,x = dtype_and_x
    helpers.test_frontend_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
        'torch',
        'cos',
        input = np.asarray(x,dtype=input_dtype),
        out = None,

    )

#sin
@given(
        dtype_and_x=helpers.dtype_and_values(
            tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_torch.valid_float_dtypes)
                )
            )
        ),
        as_variable = st.booleans(),
        with_out = st.booleans(),
        num_positional_args = helpers.num_positional_args(fn_name="functional.frontends.torch.sin"),
        native_array=st.booleans(),

)
def test_torch_sin(
        dtype_and_x,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
):
    input_dtype,x = dtype_and_x
    helpers.test_frontend_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
        'torch',
        'sin',
        input = np.asarray(x,dtype=input_dtype),
        out = None,

    )