"""Collection of tests for unified neural network layers."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@given(
    dtype_x_normidxs=helpers.dtype_values_axis(
        available_dtypes=ivy_np.valid_float_dtypes,
        allow_inf=False,
        min_num_dims=1,
        min_axis=1,
        ret_shape=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="layer_norm"),
    scale=helpers.floats(min_value=0.0),
    offset=helpers.floats(min_value=0.0),
    epsilon=helpers.floats(min_value=ivy._MIN_BASE, max_value=0.1),
    new_std=helpers.floats(min_value=0.0, exclude_min=True),
    data=st.data(),
)
@handle_cmd_line_args
def test_layer_norm(
    *,
    dtype_x_normidxs,
    num_positional_args,
    scale,
    offset,
    epsilon,
    new_std,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, normalized_idxs = dtype_x_normidxs
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="layer_norm",
        x=np.asarray(x, dtype=dtype),
        normalized_idxs=normalized_idxs,
        epsilon=epsilon,
        scale=scale,
        offset=offset,
        new_std=new_std,
    )
