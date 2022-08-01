#global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch

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
        num_positional_args = helpers.num_positional_args(fn_name="functional.frontends.torch.sigmoid"),
        native_array=st.booleans(),
)
def test_torch_sigmoid(
        dtype_and_x,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
):
    input_dtype,x = dtype_and_x

    if input_dtype == 'float16':
        return

    helpers.test_frontend_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
        'torch',
        'sigmoid',
        input=np.asarray(x, dtype=input_dtype),
        out=None,

    )