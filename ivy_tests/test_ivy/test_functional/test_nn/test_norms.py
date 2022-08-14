"""Collection of tests for unified neural network layers."""

# global
import pytest
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
    scale=st.floats(min_value=0.0),
    offset=st.floats(min_value=0.0),
    epsilon=st.floats(min_value=ivy._MIN_BASE, max_value=0.1),
    new_std=st.floats(min_value=0.0, exclude_min=True),
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


@pytest.mark.parametrize(
    "x_n_ni_n_s_n_o_n_res",
    [
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            -1,
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-0.22473562, 2.0, 6.6742067], [-0.8989425, 5.0, 13.348413]],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array])
def test_layer_norm_ground_truth(x_n_ni_n_s_n_o_n_res, dtype, tensor_fn, device, call):
    # smoke test
    x, norm_idxs, scale, offset, true_res = x_n_ni_n_s_n_o_n_res
    x = tensor_fn(x, dtype=dtype, device=device)
    scale = tensor_fn(scale, dtype=dtype, device=device)
    offset = tensor_fn(offset, dtype=dtype, device=device)
    true_res = tensor_fn(true_res, dtype=dtype, device=device)
    ret = ivy.layer_norm(x, norm_idxs, scale=scale, offset=offset)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(
        call(ivy.layer_norm, x, norm_idxs, scale=scale, offset=offset),
        ivy.to_numpy(true_res),
    )
    # compilation test
    if call in [helpers.torch_call]:
        # this is not a backend implemented function
        return
    helpers.assert_compilable(ivy.layer_norm)
