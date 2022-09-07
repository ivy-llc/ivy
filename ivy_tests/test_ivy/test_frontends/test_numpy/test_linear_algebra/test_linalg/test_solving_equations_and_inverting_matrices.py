# global
import numpy as np
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# inv


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_dim_size=6,
        max_dim_size=6,
        min_num_dims=2,
        max_num_dims=2,
    ).filter(lambda x: np.linalg.det(x[1]) != 0),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.inv"
    ),
)
def test_numpy_inv(dtype_and_x, as_variable, native_array, num_positional_args, fw):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.inv",
        a=np.array(x, dtype=dtype),
    )


# pinv


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.pinv"
    ),
)
def test_numpy_pinv(dtype_and_x, as_variable, native_array, num_positional_args, fw):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.pinv",
        a=np.array(x, dtype=dtype),
        rtol=1e-15,
    )
