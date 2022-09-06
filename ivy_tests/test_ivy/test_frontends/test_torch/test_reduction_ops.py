import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.statistical_dtype_values(function="dist"),
    dtype_and_y=helpers.statistical_dtype_values(function="dist"),
    as_variable=helpers.array_bools(num_arrays=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.dist"
    ),
    native_array=helpers.array_bools(num_arrays=2),
    p=st.integers(),
)
def test_torch_dist(
    dtype_and_x,
    dtype_and_y,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    p,
):
    input_x_dtype, x, dim_y = dtype_and_x
    input_y_dtype, y, dim_y = dtype_and_y
    helpers.test_frontend_function(
        input_dtypes=[input_x_dtype, input_y_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw="torch",
        frontend="torch",
        fn_tree="dist",
        input=np.asarray(x, dtype=input_x_dtype),
        other=np.asarray(y, dtype=input_y_dtype),
        p=p,
    )
