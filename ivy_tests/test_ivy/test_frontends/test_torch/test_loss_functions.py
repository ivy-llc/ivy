# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


# binary_cross_entropy
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ).difference(("float64",))
        ),
        num_arrays=3,
        min_value=-1.0,
        max_value=1.0,
        allow_inf=False,
        shape=helpers.get_shape(
                          allow_none = False, min_num_dims = 1,
                          max_num_dims = 5, min_dim_size = 2
                                   )
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["mean", "none", "sum", None]),
    
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.nn.binary_cross_entropy"
    ),
    native_array=st.booleans(),
)
def test_torch_binary_cross_entropy(
    dtype_and_x,
    size_average, 
    reduce, 
    reduction,
    
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_name="nn.functional.binary_cross_entropy",
        rtol=1e-04,
        
        input=np.random.rand(*(np.asarray(x[0]).shape)).astype(input_dtype[0]),
        target=np.asarray(x[1], dtype=input_dtype[1]),
        weight=np.random.rand(np.asarray(x[2]).shape[-1]).astype(input_dtype[2]),
        size_average=size_average, 
        reduce=reduce, 
        reduction=reduction,
    )
