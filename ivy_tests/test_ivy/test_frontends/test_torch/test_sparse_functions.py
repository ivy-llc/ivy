


# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch



@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_torch.valid_int_dtypes, 
        min_value=0, 
        max_value=7
    ),
    num_classes=st.integers(
        min_value=8, 
        max_value=11
    ),
    num_positional_args = helpers.num_positional_args(
        fn_name='ivy.functional.frontends.torch.sparse_functions.one_hot')
)
def test_torch_one_hot(
    dtype_and_x, 
    num_classes,
    as_variable,
    num_positional_args,
    native_array,
    fw
):
    input_dtype, x = dtype_and_x
    
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend='torch', 
        fn_tree='nn.functional.one_hot',
        x=np.asarray(x, dtype=input_dtype),
        num_classes=num_classes
    )    
    