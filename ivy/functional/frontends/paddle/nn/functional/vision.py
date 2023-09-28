# local

import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.utils.assertions import check_equal


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


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def channel_shuffle(x, groups, data_format="NCHW", name=None):
    if len(ivy.shape(x)) != 4:
        raise ValueError(
            "Input x should be 4D tensor, but received x with the shape of {}".format(
                ivy.shape(x)
            )
        )

    if not isinstance(groups, int):
        raise TypeError("groups must be int type")

    if groups <= 0:
        raise ValueError("groups must be positive")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'."
            "But recevie Attr(data_format): {} ".format(data_format)
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


"Add NN Vision Functions to Paddle Frontend "

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros'):
    """
    Samples elements from the input tensor using bilinear or nearest-neighbor sampling.
    :param input: The input tensor of shape (batch_size, channels, height, width).
    :param grid: The sampling grid of shape (batch_size, height, width, 2).
    :param mode: The sampling mode - 'bilinear' or 'nearest'. Default is 'bilinear'.
    :param padding_mode: The padding mode when grid values are out-of-bounds. Supports 'zeros' and 'border'.
    :return: The sampled output tensor.
    """

    # Bilinear sampling
    if mode == 'bilinear':
        # Extract dimensions
        B, C, H, W = input.shape
        _, H_prime, W_prime, _ = grid.shape

        # Normalize the grid values to be in the range [-1, 1]
        grid = 2.0 * grid / torch.tensor([W - 1, H - 1], dtype=torch.float32) - 1.0

        # Map grid points to pixel indices
        grid = (grid + 1) * torch.tensor([W - 1, H - 1], dtype=torch.float32) / 2
        grid_floor = torch.floor(grid).long()
        grid_ceil = grid_floor + 1

        # Get pixel values at grid points
        indices_tl = grid_floor[..., 1, :, :].clamp(0, H - 1), grid_floor[..., 0, :, :].clamp(0, W - 1)
        indices_tr = grid_floor[..., 1, :, :].clamp(0, H - 1), grid_ceil[..., 0, :, :].clamp(0, W - 1)
        indices_bl = grid_ceil[..., 1, :, :].clamp(0, H - 1), grid_floor[..., 0, :, :].clamp(0, W - 1)
        indices_br = grid_ceil[..., 1, :, :].clamp(0, H - 1), grid_ceil[..., 0, :, :].clamp(0, W - 1)

        values_tl = input[..., indices_tl[0], indices_tl[1]]
        values_tr = input[..., indices_tr[0], indices_tr[1]]
        values_bl = input[..., indices_bl[0], indices_bl[1]]
        values_br = input[..., indices_br[0], indices_br[1]]

        # Calculate bilinear interpolation weights
        wa = ((grid[..., 0, :, :] - indices_tl[1].float()) * (grid[..., 1, :, :] - indices_tl[0].float())).unsqueeze(1)
        wb = ((indices_tr[1].float() - grid[..., 0, :, :]) * (grid[..., 1, :, :] - indices_tr[0].float())).unsqueeze(1)
        wc = ((grid[..., 0, :, :] - indices_bl[1].float()) * (indices_bl[0].float() - grid[..., 1, :, :])).unsqueeze(1)
        wd = ((indices_br[1].float() - grid[..., 0, :, :]) * (indices_br[0].float() - grid[..., 1, :, :])).unsqueeze(1)

        output = wa * values_tl + wb * values_tr + wc * values_bl + wd * values_br

    # Nearest-neighbor sampling
    elif mode == 'nearest':
        # Round the grid values to get the closest integer indices
        x_rounded = torch.round(grid[..., 0]).long()
        y_rounded = torch.round(grid[..., 1]).long()

        if padding_mode == 'zeros':
            # Create masks for out-of-bound x and y positions
            mask_x = torch.logical_or(x_rounded < 0, x_rounded >= W)
            mask_y = torch.logical_or(y_rounded < 0, y_rounded >= H)

            # Using the indices, gather the values from the input tensor
            sampled_output = input[..., y_rounded, x_rounded]

            # Use the mask to set out-of-bound positions in the output to zero
            sampled_output = torch.where(mask_x | mask_y, torch.zeros_like(sampled_output), sampled_output)

        elif padding_mode == 'border':
            # Clamp the indices to lie within the borders
            x_clamped = torch.clamp(x_rounded, 0, W - 1)
            y_clamped = torch.clamp(y_rounded, 0, H - 1)

            # Using the clamped indices, gather the values from the input tensor
            sampled_output = input[..., y_clamped, x_clamped]

        else:
            raise ValueError("Unsupported padding_mode. Expected 'zeros' or 'border'.")

    else:
        raise ValueError("Unsupported mode. Expected 'bilinear' or 'nearest'.")

    return sampled_output

