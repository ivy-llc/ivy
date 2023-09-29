# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.torch.nn.functional import convolution_functions


# --- Helpers --- #
# --------------- #


def _channel_first_input(x, data_format):
    ndims = len(x.shape)
    dims = ndims - 2
    if 1 >= dims >= 3:
        raise ivy.utils.exceptions.IvyError(f"invalid for input with {dims} dims")
    # channel first input
    if data_format not in ["NCL", "NCHW", "NCDHW"]:
        if data_format in ["NLC", "NHWC", "NDHWC"]:
            x = ivy.permute_dims(x, axes=(0, ndims - 1, *range(1, ndims - 1)))
        else:
            raise ivy.utils.exceptions.IvyError(
                "data_format should be " + "'NCL' or 'NLC' "
                if dims == 1
                else (
                    "'NCHW' or 'NHWC' "
                    if dims == 2
                    else "'NCDHW' or 'NDHWC' " + f"but got data_format '{data_format}'"
                )
            )
    return x


def _conv(
    x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format="NLC"
):
    x = _channel_first_input(x, data_format)

    if padding == "same":
        dilation = 1
    return convolution_functions._conv(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _conv_transpose(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups=1,
    data_format="NLC",
):
    x = _channel_first_input(x, data_format)
    if padding == "same":
        dilation = 1
    return convolution_functions._conv_transpose(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


# --- Main --- #
# ------------ #


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv1d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NCL",
    name=None,
):
    return _conv(x, weight, bias, stride, padding, dilation, groups, data_format)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv1d_transpose(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
    output_size=None,
    data_format="NCL",
    name=None,
):
    return _conv_transpose(
        x, weight, bias, stride, padding, output_padding, dilation, groups, data_format
    )


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv2d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NCHW",
    name=None,
):
    return _conv(x, weight, bias, stride, padding, dilation, groups, data_format)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv2d_transpose(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups=1,
    output_size=None,
    data_format="NCHW",
    name=None,
):
    return _conv_transpose(
        x, weight, bias, stride, padding, output_padding, dilation, groups, data_format
    )


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv3d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NCDHW",
    name=None,
):
    return _conv(x, weight, bias, stride, padding, dilation, groups, data_format)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv3d_transpose(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
    output_size=None,
    data_format="NCDHW",
    name=None,
):
    return _conv_transpose(
        x, weight, bias, stride, padding, output_padding, dilation, groups, data_format
    )


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64")},
    "paddle",
)
@inputs_to_ivy_arrays
def cross_entropy(
    input,
    label,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    soft_label=False,
    axis=-1,
    use_softmax=True,
    name=None,
):
    result = ivy.cross_entropy(label, input)

    if soft_label:
        if use_softmax:
            # Soft labels with softmax
            pass
        else:
            # Soft labels without softmax
            pass
    else:
        if use_softmax:
            # Hard labels with softmax
            pass
        else:
            # Hard labels without softmax
            pass

    if weight is not None:
        if soft_label:
            # Apply weight to soft labels
            pass
        else:
            # Apply weight to hard labels
            pass

    # Process reduction
    if reduction == "none":
        return result
    elif reduction == "sum":
        return ivy.sum(result)
    elif reduction == "mean":
        if weight is None:
            return ivy.mean(result)
        else:
            if soft_label:
                # Weighted mean for soft labels
                pass
            else:
                # Weighted mean for hard labels
                pass


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def dice_loss(input, label, epsilon=0.00001, name=None):