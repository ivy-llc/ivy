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
    num_positional_args_init=helpers.num_positional_args(fn_name="LayerNorm.__init__"),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="LayerNorm._forward"
    ),
)
# @pytest.mark.parametrize(
#     "x_n_ns_n_target",
#     [
#         (
#             [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
#             [3],
#             [[-1.2247356, 0.0, 1.2247356], [-1.2247356, 0.0, 1.2247356]],
#         ),
#     ],
# )
# @pytest.mark.parametrize("with_v", [True, False])
# @pytest.mark.parametrize("dtype", ["float32"])
# @pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_layer_norm_layer(
    *,
    dtype_and_x,
    new_std,
    init_with_v,
    method_with_v,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
    device,
):
    input_dtype, x = dtype_and_x
    x = np.asarray(x, dtype=input_dtype)
    shape = x.shape
    print(f"\nbefore -: {input_dtype}")
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        num_positional_args_method=num_positional_args_method,
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
