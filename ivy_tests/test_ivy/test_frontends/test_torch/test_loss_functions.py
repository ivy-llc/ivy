# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


# cross_entropy
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size= 1,
    ),
    dtype_and_target=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        min_value=1.0013580322265625e-05,
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.cross_entropy"
    ),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
)
def test_torch_cross_entropy(
    dtype_and_input, dtype_and_target, as_variable, num_positional_args, native_array, fw
):
    inputs_dtype, input = dtype_and_input
    target_dtype, target = dtype_and_target
    helpers.test_frontend_function(
        input_dtypes=[inputs_dtype, target_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.cross_entropy",
        input=np.asarray(input, dtype=inputs_dtype),
        target=np.asarray(target, dtype=target_dtype),
    )
