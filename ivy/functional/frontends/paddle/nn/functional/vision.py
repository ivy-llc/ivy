# local

import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.utils.assertions import check_equal


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
def affine_grid(theta, out_shape, align_corners=True):
    if len(out_shape) == 4:
        N, C, H, W = out_shape
        base_grid = ivy.empty((N, H, W, 3))
        if align_corners:
            base_grid[:, :, :, 0] = ivy.linspace(-1, 1, W)
            base_grid[:, :, :, 1] = ivy.expand_dims(ivy.linspace(-1, 1, H), axis=-1)
            height_values = ivy.expand_dims(ivy.linspace(-1, 1, H), axis=-1)
            base_grid[:, :, :, 1] = ivy.array(
                [[[height_values[i]] * W for i in range(H)]]
            )[:, :, :, 0]
            base_grid[:, :, :, 2] = ivy.full((H, W), 1)
            grid = ivy.matmul(base_grid.view((N, H * W, 3)), theta.swapaxes(1, 2))
            return grid.view((N, H, W, 2))
        else:
            base_grid[:, :, :, 0] = ivy.linspace(-1, 1, W) * (W - 1) / W
            base_grid[:, :, :, 1] = ivy.expand_dims(
                ivy.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            height_values = ivy.expand_dims(
                ivy.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            base_grid[:, :, :, 1] = ivy.array(
                [[[height_values[i]] * W for i in range(H)]]
            )[:, :, :, 0]
            base_grid[:, :, :, 2] = ivy.full((H, W), 1)
        grid = ivy.matmul(base_grid.view((N, H * W, 3)), ivy.swapaxes(theta, 1, 2))
        return grid.view((N, H, W, 2))
    else:
        N, C, D, H, W = out_shape
        base_grid = ivy.empty((N, D, H, W, 4))
        if align_corners:
            base_grid[:, :, :, :, 0] = ivy.linspace(-1, 1, W)
            base_grid[:, :, :, :, 1] = ivy.expand_dims(ivy.linspace(-1, 1, H), axis=-1)
            height_values = ivy.linspace(-1, 1, H)
            base_grid[:, :, :, :, 1] = ivy.array(
                [[[[height_values[i]] * W for i in range(H)]] * D]
            )
            base_grid[:, :, :, :, 2] = ivy.expand_dims(
                ivy.expand_dims(ivy.linspace(-1, 1, D), axis=-1), axis=-1
            )
            width_values = ivy.linspace(-1, 1, D)
        else:
            base_grid[:, :, :, :, 0] = ivy.linspace(-1, 1, W) * (W - 1) / W
            base_grid[:, :, :, :, 1] = ivy.expand_dims(
                ivy.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            height_values = ivy.linspace(-1, 1, H) * (H - 1) / H
            base_grid[:, :, :, :, 1] = ivy.array(
                [[[[height_values[i]] * W for i in range(H)]] * D]
            )
            base_grid[:, :, :, :, 2] = ivy.expand_dims(
                ivy.expand_dims(ivy.linspace(-1, 1, D) * (D - 1) / D, axis=-1), axis=-1
            )
            width_values = ivy.linspace(-1, 1, D) * (D - 1) / D

        base_grid[:, :, :, :, 2] = ivy.array(
            [[ivy.array([[width_values[i]] * W] * H) for i in range(D)]]
        )
        base_grid[:, :, :, :, 3] = ivy.full((D, H, W), 1)
        grid = ivy.matmul(base_grid.view((N, D * H * W, 4)), theta.swapaxes(1, 2))
        return grid.view((N, D, H, W, 3))


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def channel_shuffle(x, groups, data_format="NCHW", name=None):
    if len(ivy.shape(x)) != 4:
        raise ValueError(
            "Input x should be 4D tensor, but received x with the shape of"
            f" {ivy.shape(x)}"
        )

    if not isinstance(groups, int):
        raise TypeError("groups must be int type")

    if groups <= 0:
        raise ValueError("groups must be positive")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'.But receive"
            f" Attr(data_format): {data_format} "
        )

    if data_format == "NCHW":
        b, c, h, w = ivy.shape(x)
        x = ivy.reshape(x, (b, groups, c // groups, h, w))
        x = ivy.permute_dims(x, (0, 2, 1, 3, 4))
        x = ivy.reshape(x, (b, c, h, w))
    else:
        b, h, w, c = ivy.shape(x)
        x = ivy.reshape(x, (b, h, w, groups, c // groups))
        x = ivy.permute_dims(x, (0, 1, 2, 4, 3))
        x = ivy.reshape(x, (b, h, w, c))
    return x


@to_ivy_arrays_and_back
def pixel_shuffle(x, upscale_factor, data_format="NCHW"):
    input_shape = ivy.shape(x)
    check_equal(
        len(input_shape),
        4,
        message=f"pixel shuffle requires a 4D input, but got input size {input_shape}",
    )

    if not isinstance(upscale_factor, int):
        raise TypeError("upscale factor must be int type")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'.But receive"
            f" Attr(data_format): {data_format} "
        )

    b = input_shape[0]
    c = input_shape[1] if data_format == "NCHW" else input_shape[3]
    h = input_shape[2] if data_format == "NCHW" else input_shape[1]
    w = input_shape[3] if data_format == "NCHW" else input_shape[2]

    upscale_factor_squared = upscale_factor**2

    check_equal(
        c % upscale_factor_squared,
        0,
        message=(
            "pixel shuffle expects input channel to be divisible by square of upscale"
            f" factor, but got input with sizes {input_shape}, upscale"
            f" factor={upscale_factor}, and self.size(1)={c}, is not divisible by"
            f" {upscale_factor_squared}"
        ),
        as_array=False,
    )

    oc = int(c / upscale_factor_squared)
    oh = h * upscale_factor
    ow = w * upscale_factor

    if data_format == "NCHW":
        input_reshaped = ivy.reshape(x, (b, oc, upscale_factor, upscale_factor, h, w))
    else:
        input_reshaped = ivy.reshape(x, (b, h, w, upscale_factor, upscale_factor, oc))

    if data_format == "NCHW":
        return ivy.reshape(
            ivy.permute_dims(input_reshaped, (0, 1, 4, 2, 5, 3)), (b, oc, oh, ow)
        )
    return ivy.reshape(
        ivy.permute_dims(input_reshaped, (0, 1, 4, 2, 5, 3)), (b, oh, ow, oc)
    )


@to_ivy_arrays_and_back
def pixel_unshuffle(x, downscale_factor, data_format="NCHW"):
    if len(ivy.shape(x)) != 4:
        raise ValueError(
            "Input x should be 4D tensor, but received x with the shape of"
            f" {ivy.shape(x)}"
        )

    if not isinstance(downscale_factor, int):
        raise TypeError("Downscale factor must be int type")

    if downscale_factor <= 0:
        raise ValueError("Downscale factor must be positive")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'.But receive"
            f" Attr(data_format): {data_format} "
        )

    if data_format == "NCHW":
        b, c, h, w = ivy.shape(x)
        oc = c * downscale_factor**2
        oh = h // downscale_factor
        ow = w // downscale_factor

        x = ivy.reshape(x, (b, c, oh, downscale_factor, ow, downscale_factor))
        x = ivy.permute_dims(x, (0, 1, 3, 5, 2, 4))
        x = ivy.reshape(x, (b, oc, oh, ow))
    else:
        b, h, w, c = ivy.shape(x)
        oc = c * downscale_factor**2
        oh = h // downscale_factor
        ow = w // downscale_factor

        x = ivy.reshape(x, (b, downscale_factor, oh, downscale_factor, ow, c))
        x = ivy.permute_dims(x, (0, 1, 3, 5, 2, 4))
        x = ivy.reshape(x, (b, oh, ow, oc))
    return x
