# local

import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.utils.assertions import check_equal


@to_ivy_arrays_and_back
def pixel_shuffle(x, upscale_factor, data_format="NCHW"):
    input_shape = ivy.shape(x)
    check_equal(
        len(input_shape),
        4,
        message="pixel shuffle requires a 4D input, but got input size {}".format(
            input_shape
        ),
    )

    if not isinstance(upscale_factor, int):
        raise ValueError("upscale factor must be int type")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'."
            "But recevie Attr(data_format): {} ".format(data_format)
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
            " factor, but got input with sizes {}, upscale factor={}, and"
            " self.size(1)={}, is not divisible by {}".format(
                input_shape, upscale_factor, c, upscale_factor_squared
            )
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
def grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True):

    _modes = ['bilinear', 'nearest']
    _padding_modes = ['zeros', 'reflection', 'border']
    if mode not in _modes:
        raise ValueError(
            "The mode of grid sample function should be in {}, but got: {}".format(
                _modes, mode
            )
        )
    if padding_mode not in _padding_modes:
        raise ValueError(
            "The padding mode of grid sample function should be in {}, but got: {}".format(
                _padding_modes, padding_mode
            )
        )

    if not isinstance(align_corners, bool):
        raise ValueError(
            "The align corners should be bool, but got: {}".format(
                align_corners
            )
        )
    N, C, H, W = x.shape
    if len(grid.shape) == 4:
        _, grid_H, grid_W, _ = grid.shape
    elif len(grid.shape) == 5:
        _, _, grid_D, grid_H, grid_W = grid.shape
    else:
        raise ValueError("Invalid grid shape")

    grid_x = grid[:, :, :, 0]
    grid_y = grid[:, :, :, 1]

    if align_corners:
        grid_x = 0.5 * (grid_x + 1) * (W - 1)
        grid_y = 0.5 * (grid_y + 1) * (H - 1)
    else:
        grid_x = 0.5 * grid_x * (W - 1)
        grid_y = 0.5 * grid_y * (H - 1)

    y_t = ivy.zeros((N, C, grid_H, grid_W), dtype=x.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(grid_H):
                for j in range(grid_W):
                    x_float = grid_x[n, i, j]
                    y_float = grid_y[n, i, j]

                    x_w = int(ivy.floor(x_float))
                    x_e = x_w + 1
                    y_n = int(ivy.floor(y_float))
                    y_s = y_n + 1

                    d_w = x_float - x_w
                    d_e = x_e - x_float
                    d_n = y_float - y_n
                    d_s = y_s - y_float

                    if mode == 'bilinear':
                        wn = x[n, c, y_n, x_w]
                        en = x[n, c, y_n, x_e]
                        ws = x[n, c, y_s, x_w]
                        es = x[n, c, y_s, x_e]

                        if padding_mode == 'border':
                            wn = x[n, c, max(y_n, 0), max(x_w, 0)]
                            en = x[n, c, max(y_n, 0), min(x_e, W - 1)]
                            ws = x[n, c, min(y_s, H - 1), max(x_w, 0)]
                            es = x[n, c, min(y_s, H - 1), min(x_e, W - 1)]
                        elif padding_mode == 'reflection':
                            wn = x[n, c, H - 1 - abs(y_n), W - 1 - abs(x_w)]
                            en = x[n, c, H - 1 - abs(y_n), W - 1 - abs(x_e)]
                            ws = x[n, c, H - 1 - abs(y_s), W - 1 - abs(x_w)]
                            es = x[n, c, H - 1 - abs(y_s), W - 1 - abs(x_e)]
                        elif padding_mode == 'zeros':
                            wn = 0 if y_n < 0 or x_w < 0 else x[n, c, y_n, x_w]
                            en = 0 if y_n < 0 or x_e >= W else x[n, c, y_n, x_e]
                            ws = 0 if y_s >= H or x_w < 0 else x[n, c, y_s, x_w]
                            es = 0 if y_s >= H or x_e >= W else x[n, c, y_s, x_e]

                        value = wn * d_e * d_s + en * d_w * d_s + ws * d_e * d_n + es * d_w * d_n
                    elif mode == 'nearest':
                        x_int = int(ivy.round(x_float))
                        y_int = int(ivy.round(y_float))

                        if padding_mode == 'border':
                            x_int = ivy.clip(x_int, 0, W - 1)
                            y_int = ivy.clip(y_int, 0, H - 1)
                        elif padding_mode == 'reflection':
                            x_int = abs(x_int % (2 * W - 2))
                            y_int = abs(y_int % (2 * H - 2))
                        elif padding_mode == 'zeros':
                            x_int = max(0, min(x_int, W - 1))
                            y_int = max(0, min(y_int, H - 1))

                        value = x[n, c, y_int, x_int]

                    y_t[n, c, i, j] = value
    return y_t








@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
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
            base_grid[:, :, :, :, 2] = ivy.array(
                [[ivy.array([[width_values[i]] * W] * H) for i in range(D)]]
            )
            base_grid[:, :, :, :, 3] = ivy.full((D, H, W), 1)
            grid = ivy.matmul(base_grid.view((N, D * H * W, 4)), theta.swapaxes(1, 2))
            return grid.view((N, D, H, W, 3))
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
