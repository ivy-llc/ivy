# local

import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
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
