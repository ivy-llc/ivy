# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.torch.nn.functional.pooling_functions import (
    _broadcast_pooling_helper,
)
from ivy.func_wrapper import with_unsupported_dtypes


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool1d(x, output_size, name=None):
    return ivy.adaptive_avg_pool1d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool2d(x, output_size, data_format="NCHW", name=None):
    return ivy.adaptive_avg_pool2d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
def adaptive_avg_pool3d(x, output_size, data_format="NCHW", name=None):
    return ivy.adaptive_avg_pool3d(x, output_size)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def adaptive_max_pool2d(x, output_size, return_mask=None, name=None):
    return ivy.adaptive_max_pool2d(x, output_size)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
def avg_pool1d(
    x, kernel_size, stride=None, padding=0, exclusive=True, ceil_mode=False, name=None
):
    data_format = "NCW"
    exclusive = not exclusive
    if stride is None:
        stride = kernel_size
    kernel_size = _broadcast_pooling_helper(kernel_size, "1d", name="kernel_size")
    padding = _broadcast_pooling_helper(padding, "1d", name="padding")
    # Figure out padding string
    if all(
        [pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in zip(kernel_size, padding)]
    ):
        padding = "SAME"
    else:
        padding = "VALID"

    return ivy.avg_pool1d(
        x,
        kernel_size,
        stride,
        padding,
        count_include_pad=exclusive,
        ceil_mode=ceil_mode,
        data_format=data_format,
    )


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def avg_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    exclusive=True,
    divisor_override=None,
    data_format="NCHW",
    name=None,
):
    if stride is None:
        stride = kernel_size
    kernel_size = _broadcast_pooling_helper(kernel_size, "2d", name="kernel_size")
    padding = _broadcast_pooling_helper(padding, "2d", name="padding")
    # Figure out padding string
    if all(
        [pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in zip(kernel_size, padding)]
    ):
        padding = "SAME"
    else:
        padding = "VALID"

    count_include_pad = not exclusive
    return ivy.avg_pool2d(
        x,
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
    )


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def max_unpool1d(
    x,
    indices,
    kernel_size,
    stride=None,
    padding=0,
    data_format="NCL",
    output_size=None,
    name=None,
):
    return ivy.max_unpool1d(x, indices, kernel_size, stride, padding, data_format)

def max_pool3d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, data_format='NCDHW'):
    """
    3D max pooling operation

    Args:
        x (Tensor): The input tensor.
        kernel_size (int or tuple): The size of the pooling kernel.
        stride (int or tuple, optional): The stride of the pooling operation. Default: None.
        padding (int or tuple, optional): The padding of the pooling operation. Default: 0.
        dilation (int or tuple, optional): The dilation of the pooling operation. Default: 1.
        ceil_mode (bool, optional): Whether to use ceil mode for the pooling operation. Default: False.
        data_format (str, optional): The data format of the input tensor. Default: 'NCDHW'.

    Returns:
        Tensor: The output tensor.
    """

    if stride is None:
        stride = kernel_size

    if padding is None:
        padding = 0

    if dilation is None:
        dilation = 1

    if ceil_mode is None:
        ceil_mode = False

    if data_format is None:
        data_format = 'NCDHW'

    return F.max_pool3d(x, kernel_size, stride, padding, dilation, ceil_mode, data_format)


def max_pool3d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    name=None
):
    """3D max pooling operation.

    This function performs 3D max pooling over the input tensor.

    Args:
        x (ivy.Array): Input tensor.
        kernel_size (Union[int, Tuple[int, int, int]]): Size of the max pooling window.
        stride (Union[int, Tuple[int, int, int]], optional): Stride of the max pooling
            operation. Defaults to ``kernel_size``.
        padding (Union[int, Tuple[int, int, int]], optional): Zero-padding added to each side
            of the input. Defaults to ``0``.
        ceil_mode (bool, optional): If ``True``, use "ceil" rounding when computing the
            output shape. Defaults to ``False``.
        name (str, optional): Optional name for the operation.

    Returns:
        ivy.Array: The max pooled output tensor.
    """

    # type-checking

    backend = current_backend()
    _ = backend.inputs_to_ivy_arrays([x])
    _ = backend.as_native_dtype(kernel_size)
    _ = backend.as_native_dtype(stride)
    _ = backend.as_native_dtype(padding)
    _ = backend.as_native_dtype(ceil_mode)
    _ = backend.as_native_dtype(name)

    # backend-specific implementation

    if backend.framework == "tensorflow":
        return _max_pool3d_tensorflow(x, kernel_size, stride, padding, ceil_mode, name)
    elif backend.framework == "torch":
        return _max_pool3d_torch(x, kernel_size, stride, padding, ceil_mode, name)
    elif backend.framework == "jax":
        return _max_pool3d_jax(x, kernel_size, stride, padding, ceil_mode, name)
    elif backend.framework == "numpy":
        return _max_pool3d_numpy(x, kernel_size, stride, padding, ceil_mode, name)
