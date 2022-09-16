"""Collection of tests for normalization layers."""

# global
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# layer norm
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    new_std=st.floats(min_value=0.0, max_value=1.0),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_layer_norm_layer(
    *,
    dtype_and_x,
    new_std,
    init_with_v,
    method_with_v,
    as_variable,
    native_array,
    container,
    fw,
    device,
):
    input_dtype, x = dtype_and_x
    x = np.asarray(x, dtype=input_dtype)
    shape = x.shape
    helpers.test_method(
        num_positional_args_init=1,
        num_positional_args_method=5,
        all_as_kwargs_np_init={
            "normalized_shape": shape,
            "epsilon": ivy._MIN_BASE,
            "elementwise_affine": True,
            "new_std": new_std,
            "device": device,
            "dtype": input_dtype,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        container_flags_method=container,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        all_as_kwargs_np_method={"inputs": x},
        fw=fw,
        class_name="LayerNorm",
    )
