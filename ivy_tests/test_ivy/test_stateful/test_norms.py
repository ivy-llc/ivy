"""Collection of tests for normalization layers."""

# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method


@handle_method(
    method_tree="LayerNorm.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=2
    ),
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
    test_gradients,
    on_device,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "normalized_shape": x[0].shape,
            "eps": ivy.min_base,
            "elementwise_affine": True,
            "new_std": new_std,
            "device": on_device,
            "dtype": input_dtype[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"inputs": x[0]},
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_gradients=test_gradients,
        on_device=on_device,
    )


@st.composite
def _generate_batchnorm_data(draw):
    batch_size = draw(st.integers(min_value=2, max_value=5))
    num_features = draw(st.integers(min_value=2, max_value=3))
    num_dims = draw(st.integers(min_value=1, max_value=3))
    dims = [draw(st.integers(1, 5)) for i in range(num_dims)]
    x_shape = [batch_size] + [*dims] + [num_features]
    dtype, inputs = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float", full=True),
            shape=x_shape,
            min_value=0,
            max_value=1,
        ).filter(lambda x: x[0][0] not in ["float64"])
    )
    return dtype, inputs, num_features


@handle_method(
    method_tree="BatchNorm2D.__call__",
    dtype_and_x_features=_generate_batchnorm_data(),
    momentum=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_batch_norm_2d_layer(
    *,
    dtype_and_x_features,
    momentum,
    init_with_v,
    method_with_v,
    test_gradients,
    on_device,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    input_dtype, x, features = dtype_and_x_features
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "num_features": features,
            "eps": ivy.min_base,
            "affine": True,
            "momentum": momentum,
            "track_running_stats": True,
            "device": on_device,
            "dtype": input_dtype[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"inputs": x[0]},
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_gradients=test_gradients,
        rtol_=1e-02,
        atol_=1e-02,
        on_device=on_device,
    )
