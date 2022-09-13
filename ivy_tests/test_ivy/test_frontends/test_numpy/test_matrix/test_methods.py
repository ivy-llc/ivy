# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# argmax
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    as_variable=helpers.array_bools(num_arrays=1),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.argmax"
    ),
    native_array=helpers.array_bools(num_arrays=1),
    keep_dims=st.booleans(),
)
def test_numpy_argmax(
    dtype_x_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    input_dtype = [input_dtype]
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="argmax",
        a=np.asarray(x, dtype=input_dtype[0]),
        axis=axis,
        out=None,
        keepdims=keep_dims,
        test_values=False,
    )
