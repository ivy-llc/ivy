# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


# cross_entropy
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=1,
    ),
    dtype_and_target=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=1,
    ),
    dtype_and_weights=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        shape=(1,),
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["mean", "none", "sum"]),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.cross_entropy"
    ),
)
def test_torch_cross_entropy(
    dtype_and_input,
    dtype_and_target,
    dtype_and_weights,
    size_average,
    reduce,
    reduction,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    inputs_dtype, input = dtype_and_input
    target_dtype, target = dtype_and_target
    weights_dtype, weights = dtype_and_weights
    helpers.test_frontend_function(
        input_dtypes=[inputs_dtype, target_dtype, weights_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.cross_entropy",
        input=np.asarray(input, dtype=inputs_dtype),
        target=np.asarray(target, dtype=target_dtype),
        weight=np.asarray(weights, dtype=weights_dtype),
        size_average=size_average,
        ignore_index=-100,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=0.0,
    )
