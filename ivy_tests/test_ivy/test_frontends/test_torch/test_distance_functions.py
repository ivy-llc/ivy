# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.torch as ivy_torch
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x1=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            )
        ),
        # TODO: Find a better way to make sure that x1 and x2 are the same size
        # This does ensure that, but it also ensures that they are always 3x3 matrices
        max_dim_size=3,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=3,
    ),
    dtype_and_x2=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            )
        ),
        # TODO: Find a better way to make sure that x1 and x2 are the same size
        # This does ensure that, but it also ensures that they are always 3x3 matrices
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=3,
        max_dim_size=3,
    ),
    # The following bounds are arbitrary (other than min_value = 0 for p)
    p=helpers.floats(
        min_value=0, max_value=10.0, allow_nan=False, allow_inf=False, exclude_min=True
    ),
    eps=helpers.floats(
        min_value=1e-8, max_value=1e-4, allow_nan=False, allow_inf=False
    ),
    keepdims=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.pairwise_distance"
    ),
    as_variable=st.booleans(),
    native_array=st.booleans(),
)
def test_torch_pairwise_distance(
    dtype_and_x1,
    dtype_and_x2,
    p,
    eps,
    keepdims,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    x1_dtype, x1 = dtype_and_x1
    x2_dtype, x2 = dtype_and_x2
    helpers.test_frontend_function(
        input_dtypes=[x1_dtype, x2_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="pairwise_distance",
        x1=np.asarray(x1, dtype=x1_dtype),
        x2=np.asarray(x2, dtype=x1_dtype),
        p=p,
        keepdim=keepdims,
    )
