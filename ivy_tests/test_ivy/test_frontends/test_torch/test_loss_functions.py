# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch
import ivy_tests.test_ivy.helpers as helpers


@given(
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ).difference(("float64",))
        ), 
        min_value=0, 
        max_value=1, 
        allow_inf=False, 
        min_num_dims=1, 
        max_num_dims=1, 
        min_dim_size=2,
    ), 
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ).difference(("float64",))
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
    dtype_and_weight=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ).difference(("float64",))
        ), 
        min_value=1.0013580322265625e-05, 
        max_value=1, 
        allow_inf=False, 
        min_num_dims=1, 
        max_num_dims=1, 
        min_dim_size=2, 
    ),
    size_average=st.booleans(), 
    reduce=st.booleans(), 
    reduction=st.sampled_from(["mean", "none", "sum", None]), 
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.binary_cross_entropy"
    ),
)
def test_binary_cross_entropy(
    dtype_and_true, 
    dtype_and_pred, 
    dtype_and_weight, 
    size_average, 
    reduce, 
    reduction, 
    num_positional_args, 
    fw,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true
    weight_dtype, weight = dtype_and_weight
    
    helpers.test_frontend_function(
        input_dtypes=[pred_dtype, true_dtype, weight_dtype], 
        as_variable_flags=False, 
        with_out=False, 
        num_positional_args=num_positional_args, 
        native_array_flags=False, 
        fw=fw, 
        frontend="torch", 
        fn_tree="nn.functional.binary_cross_entropy", 
        input=np.asarray(pred, dtype=pred_dtype), 
        target=np.asarray(true, dtype=true_dtype), 
        weight=np.asarray(weight, dtype=weight_dtype), 
        size_average=size_average, 
        reduce=reduce, 
        reduction=reduction,
    )
