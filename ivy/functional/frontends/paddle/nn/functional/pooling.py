# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


def _broadcast_pooling_helper(x, pool_dims, name):
    if isinstance(x, int):
        return tuple([x for _ in range(pool_dims)])
    if len(x) == 1:
        return tuple([x[0] for _ in range(pool_dims)])
    elif len(x) == pool_dims:
        return tuple(x)
    elif isinstance(x, str) and name == "padding":
        return x
    elif len(x) != pool_dims:
        raise ValueError(
            f"`{name}` must either be a single int, or a tuple of {pool_dims} ints. "
        )


def check_neg_arr(x, kernel_size, padding, stride, dilation, data_format):
    input_size = []
    if data_format == "NHWC":
        n = x.shape[0]
        input_size.append(x.shape[1])
        input_size.append(x.shape[2])
    elif data_format == "NCHW":
        n = x.shape[0]
        input_size.append(x.shape[2])
        input_size.append(x.shape[3])
    else:
        raise ValueError("data_format value must be either NCHW or NHCW")

    if isinstance(padding, str):
        padding = [
            min(
                kernel_size[i] // 2,
                (
                    (input_size[i] - 1) * stride[i]
                    - input_size[i]
                    + 1
                    + (dilation[i] * (kernel_size[i] - 1))
                )
                / 2,
            )
            for i in range(2)
        ]

    h_out = int(
        (
            (input_size[0] + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1)) - 1)
            / stride[0]
        )
        + 1
    )
    w_out = int(
        (
            (input_size[1] + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1)) - 1)
            / stride[1]
        )
        + 1
    )

    if h_out <= 0 or w_out <= 0:
        raise RuntimeError(
            f"Given input size: ({n}x{input_size[0]}x{input_size[1]}). Calculated"
            f" output size: ({n}x{h_out}x{w_out}). Output size is too small"
        )


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
def Maxpool2d(
    x,
    kernel_size,
    stride=1,
    padding=0,
    /,
    *,
    data_format="NHWC",
    dilation=1,
    ceil_mode=False,
    out=None,
):
    kernel_size = _broadcast_pooling_helper(kernel_size, 2, name="kernel_size")
    stride = _broadcast_pooling_helper(stride, 2, name="stride")
    padding = _broadcast_pooling_helper(padding, 2, name="padding")
    dilation = _broadcast_pooling_helper(dilation, 2, name="dilation")
    kernel_pads = list(zip(kernel_size, padding))

    # Padding should be less than or equal to half of kernel size
    if not isinstance(padding, str):
        if not all([pad <= kernel // 2 for kernel, pad in kernel_pads]):
            raise ValueError(
                "pad should be smaller than or equal to half of kernel size, "
                f"but got padding={padding}, kernel_size={kernel_size}. "
            )

    # Dilation must always follows the formual
    # H(out) = H(in) + 2*Padding[0] - dilation[0]*(kernel_size[0]-1)
    #          -----------------------------------------------------   + 1
    #                           Stride[0]
    check_neg_arr(x, kernel_size, padding, stride, dilation, data_format)

    return ivy.max_pool2d(
        x,
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        dilation=dilation,
        ceil_mode=ceil_mode,
        out=out,
    )
