# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_torch.test_nn.test_functional.\
    test_convolution_functions import (
        x_and_filters,
        _output_shape
    )

# conv1d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.conv1d",
    dtype_vals=x_and_filters(dim=1),
)
def test_paddle_conv1d(
    *,
    dtype_vals,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vals, weight, bias, dilations, strides, padding, fc = dtype_vals
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=vals,
        weight=weight,
        bias=bias,
        stride=strides,
        padding=padding,
        dilation=dilations,
        groups=fc,
    )


# conv2d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.conv2d",
    dtype_vals=x_and_filters(dim=2),
)
def test_paddle_conv2d(
    *,
    dtype_vals,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vals, weight, bias, dilations, strides, padding, fc = dtype_vals
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=vals,
        weight=weight,
        bias=bias,
        stride=strides,
        padding=padding,
        dilation=dilations,
        groups=fc,
    )


# conv3d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.conv3d",
    dtype_vals=x_and_filters(dim=3),
)
def test_paddle_conv3d(
    *,
    dtype_vals,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vals, weight, bias, dilations, strides, padding, fc = dtype_vals
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=vals,
        weight=weight,
        bias=bias,
        stride=strides,
        padding=padding,
        dilation=dilations,
        groups=fc,
    )

# conv1d_transpose
@handle_frontend_test(
    fn_tree="paddle.nn.functional.conv1d_transpose",
    dtype_vals=x_and_filters(dim=1, transpose=True),
)
def test_paddle_conv1d_tranpose(
    *,
    dtype_vals,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vals, weight, bias, dilations, strides, padding, output_pad, fc = dtype_vals
    dilations = 1  # ToDo: remove this when support for dilation > 1 is added
    assume(
        all(
            x > 0
            for x in _output_shape(
                1, dilations, strides, padding, output_pad, vals.shape, weight.shape
            )
        )
    )
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=vals,
        weight=weight,
        bias=bias,
        stride=strides,
        padding=padding,
        output_padding=output_pad,
        groups=fc,
        dilation=dilations,
    )

# conv2d_transpose
@handle_frontend_test(
    fn_tree="paddle.nn.functional.conv2d_transpose",
    dtype_vals=x_and_filters(dim=2, transpose=True),
)
def test_paddle_conv2d_tranpose(
    *,
    dtype_vals,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vals, weight, bias, dilations, strides, padding, output_pad, fc = dtype_vals
    dilations = 1  # ToDo: remove this when support for dilation > 1 is added
    assume(
        all(
            x > 0
            for x in _output_shape(
                2, dilations, strides, padding, output_pad, vals.shape, weight.shape
            )
        )
    )
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=vals,
        weight=weight,
        bias=bias,
        stride=strides,
        padding=padding,
        output_padding=output_pad,
        dilation=dilations,
        groups=fc,
    )

# conv3d_transpose
@handle_frontend_test(
    fn_tree="paddle.nn.functional.conv3d_transpose",
    dtype_vals=x_and_filters(dim=3, transpose=True),
)
def test_paddle_conv3d_tranpose(
    *,
    dtype_vals,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, vals, weight, bias, dilations, strides, padding, output_pad, fc = dtype_vals
    dilations = 1  # ToDo: remove this when support for dilation > 1 is added
    assume(
        all(
            x > 0
            for x in _output_shape(
                3, dilations, strides, padding, output_pad, vals.shape, weight.shape
            )
        )
    )
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=vals,
        weight=weight,
        bias=bias,
        stride=strides,
        padding=padding,
        output_padding=output_pad,
        groups=fc,
        dilation=dilations,
    )
