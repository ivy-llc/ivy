"""Collection of tests for unified neural network layers."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_x_normidxs=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=40,
        small_abs_safety_factor=40,
        safety_factor_scale="log",
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    scale_n_offset_n_new_std=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        shape=(),
        large_abs_safety_factor=40,
        small_abs_safety_factor=40,
        safety_factor_scale="log",
    ),
    epsilon=st.floats(min_value=ivy._MIN_BASE, max_value=0.1),
    num_positional_args=helpers.num_positional_args(fn_name="layer_norm"),
)
def test_layer_norm(
    *,
    dtype_x_normidxs,
    scale_n_offset_n_new_std,
    epsilon,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, normalized_idxs = dtype_x_normidxs
    _, [scale, offset, new_std] = scale_n_offset_n_new_std
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
        rtol_=1e-1,
        atol_=1e-1,
        x=np.asarray(x, dtype=dtype),
        normalized_idxs=normalized_idxs,
        epsilon=epsilon,
        scale=scale,
        offset=offset,
        new_std=new_std,
    )
