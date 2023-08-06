# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.torch.nn.functional.convolution_functions import _conv

@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv1d(
    x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
    data_format='NCL', name=None
):
    if data_format != 'NCL':
        if data_format == 'NLC':
            ndims = len(x.shape)
            x = ivy.permute_dims(x, axes=(0, ndims - 1, *range(1, ndims - 1)))
        else:
            raise ivy.utils.exceptions.IvyError(
                "data_format should be 'NCL' or 'NLC' "
                f"but got data_format '{data_format}'"
            )
    return _conv(x, weight, bias, stride, padding, dilation, groups)

@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv2d(
    x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
    data_format='NCHW', name=None
):
    if data_format != 'NCHW':
        if data_format == 'NHWC':
            ndims = len(x.shape)
            x = ivy.permute_dims(x, axes=(0, ndims - 1, *range(1, ndims - 1)))
        else:
            raise ivy.utils.exceptions.IvyError(
                "data_format should be 'NCHW' or 'NHWC' "
                f"but got data_format '{data_format}'"
            )
    return _conv(x, weight, bias, stride, padding, dilation, groups)

@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv3d(
    x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
    data_format='NCDHW', name=None
):
    # channel first input
    if data_format != 'NCDHW':
        if data_format == 'NDHWC':
            ndims = len(x.shape)
            x = ivy.permute_dims(x, axes=(0, ndims - 1, *range(1, ndims - 1)))
        else:
            raise ivy.utils.exceptions.IvyError(
                "data_format should be 'NCDHW' or 'NDHWC' "
                f"but got data_format '{data_format}'"
            )
    return _conv(x, weight, bias, stride, padding, dilation, groups)

