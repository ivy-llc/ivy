"""Collection of tests for the pooling layers."""

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method


@handle_method(
    method_tree="MaxPool2D.__call__",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=4),
)
def test_layer_maxpool2d_layer(
    *,
    x_k_s_p,
    test_gradients,
    on_device,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    input_dtype, x, kernel_size, stride, padding = x_k_s_p
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "device": on_device,
            "dtype": input_dtype[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"inputs": x[0]},
        class_name=class_name,
        method_name=method_name,
        test_gradients=test_gradients,
        on_device=on_device,
    )


@handle_method(
    method_tree="AvgPool2D.__call__",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=4),
)
def test_layer_avgpool2d_layer(
    *,
    x_k_s_p,
    test_gradients,
    on_device,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    input_dtype, x, kernel_size, stride, padding = x_k_s_p
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "device": on_device,
            "dtype": input_dtype[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"inputs": x[0]},
        class_name=class_name,
        method_name=method_name,
        test_gradients=test_gradients,
        on_device=on_device,
    )
