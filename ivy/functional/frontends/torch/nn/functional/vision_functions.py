# global

# local
import ivy
from ivy import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


# --- Helpers --- #
# --------------- #


def _handle_padding_shape(padding, n, mode):
    padding = tuple(
        [
            (padding[i * 2], padding[i * 2 + 1])
            for i in range(int(len(padding) / 2) - 1, -1, -1)
        ]
    )
    if mode == "circular":
        padding = padding + ((0, 0),) * (n - len(padding))
    else:
        padding = ((0, 0),) * (n - len(padding)) + padding
    if mode == "circular":
        padding = tuple(list(padding)[::-1])
    return padding


# --- Main --- #
# ------------ #


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def affine_grid(theta, size, align_corners=False):
    if len(size) == 4:
        N, C, H, W = size
        base_grid = ivy.empty((N, H, W, 3))
        if align_corners:
            base_grid[:, :, :, 0] = ivy.linspace(-1, 1, W)
            base_grid[:, :, :, 1] = ivy.expand_dims(ivy.linspace(-1, 1, H), axis=-1)
            base_grid[:, :, :, 2] = ivy.full((H, W), 1)
            grid = ivy.matmul(base_grid.view((N, H * W, 3)), theta.swapaxes(1, 2))
            return grid.view((N, H, W, 2))
        else:
            base_grid[:, :, :, 0] = ivy.linspace(-1, 1, W) * (W - 1) / W
            base_grid[:, :, :, 1] = ivy.expand_dims(
                ivy.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            base_grid[:, :, :, 2] = ivy.full((H, W), 1)
        grid = ivy.matmul(base_grid.view((N, H * W, 3)), ivy.swapaxes(theta, 1, 2))
        return grid.view((N, H, W, 2))
    else:
        N, C, D, H, W = size
        base_grid = ivy.empty((N, D, H, W, 4))
        if align_corners:
            base_grid[:, :, :, :, 0] = ivy.linspace(-1, 1, W)
            base_grid[:, :, :, :, 1] = ivy.expand_dims(ivy.linspace(-1, 1, H), axis=-1)
            base_grid[:, :, :, :, 2] = ivy.expand_dims(
                ivy.expand_dims(ivy.linspace(-1, 1, D), axis=-1), axis=-1
            )
            base_grid[:, :, :, :, 3] = ivy.full((D, H, W), 1)
            grid = ivy.matmul(base_grid.view((N, D * H * W, 4)), theta.swapaxes(1, 2))
            return grid.view((N, D, H, W, 3))
        else:
            base_grid[:, :, :, :, 0] = ivy.linspace(-1, 1, W) * (W - 1) / W
            base_grid[:, :, :, :, 1] = ivy.expand_dims(
                ivy.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            base_grid[:, :, :, :, 2] = ivy.expand_dims(
                ivy.expand_dims(ivy.linspace(-1, 1, D) * (D - 1) / D, axis=-1), axis=-1
            )
            base_grid[:, :, :, :, 3] = ivy.full((D, H, W), 1)
            grid = ivy.matmul(base_grid.view((N, D * H * W, 4)), theta.swapaxes(1, 2))
            return grid.view((N, D, H, W, 3))


def bicubic_interp(x, t, alpha=-0.75):
    n, h, w = t.shape
    coeffs = []
    coeffs.append(ivy.reshape(cubic_conv2(alpha, t + 1), (n, 1, h, w)))
    coeffs.append(ivy.reshape(cubic_conv1(alpha, t), (n, 1, h, w)))
    coeffs.append(ivy.reshape(cubic_conv1(alpha, 1 - t), (n, 1, h, w)))
    coeffs.append(ivy.reshape(cubic_conv2(alpha, 2 - t), (n, 1, h, w)))
    return x[0] * coeffs[0] + x[1] * coeffs[1] + x[2] * coeffs[2] + x[3] * coeffs[3]


def cubic_conv1(A, x):
    return ((A + 2) * x - (A + 3)) * x * x + 1


def cubic_conv2(A, x):
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


@with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    input_clone = ivy.copy_array(input)
    grid_clone = ivy.copy_array(grid)

    if ivy.get_num_dims(input_clone) == 4:  # sample from 2D images
        n, c, h, w = input_clone.shape
        n, to_h, to_w, gc = grid_clone.shape

        # Un-normalize 2D grid
        if align_corners:  # to range[0, size - 1]
            grid_clone[..., 0] = ((grid_clone[..., 0] + 1) / 2) * (w - 1)
            grid_clone[..., 1] = ((grid_clone[..., 1] + 1) / 2) * (h - 1)

        elif not align_corners:  # to range[0.5, size - 0.5]
            grid_clone[..., 0] = ((grid_clone[..., 0] + 1) * w - 1) / 2
            grid_clone[..., 1] = ((grid_clone[..., 1] + 1) * h - 1) / 2

        batch_coor = ivy.reshape(ivy.arange(n), (-1, 1))
        batch_coor = ivy.repeat(batch_coor, to_h * to_w, axis=1)
        batch_coor = ivy.reshape(batch_coor, (n, to_h, to_w))
        padding = [(0, 0) for _ in range(2)] + [(4, 4) for _ in range(2)]
        input_clone = ivy.pad(input_clone, padding, mode="constant", constant_values=0)

        if mode == "bicubic":
            grid_floor = ivy.floor(grid_clone)
            distance = grid_clone - grid_floor

            tx, ty = distance[..., 0], distance[..., 1]

            grid_floor -= 1
            grid_floor = [
                grid_sample_padding(
                    grid_floor + i, padding_mode, align_corners, borders=[w, h]
                )
                for i in range(4)
            ]

            w_cubic = [
                ivy.astype(grid_floor[i][..., 0] + 4, ivy.int64) for i in range(4)
            ]
            h_cubic = [
                ivy.astype(grid_floor[i][..., 1] + 4, ivy.int64) for i in range(4)
            ]

            coeffs = [
                bicubic_interp(
                    [
                        ivy.permute_dims(
                            input_clone[batch_coor, :, h_cubic[i], w_cubic[0]],
                            (0, 3, 1, 2),
                        ),
                        ivy.permute_dims(
                            input_clone[batch_coor, :, h_cubic[i], w_cubic[1]],
                            (0, 3, 1, 2),
                        ),
                        ivy.permute_dims(
                            input_clone[batch_coor, :, h_cubic[i], w_cubic[2]],
                            (0, 3, 1, 2),
                        ),
                        ivy.permute_dims(
                            input_clone[batch_coor, :, h_cubic[i], w_cubic[3]],
                            (0, 3, 1, 2),
                        ),
                    ],
                    tx,
                )
                for i in range(4)
            ]
            return bicubic_interp(coeffs, ty)

        else:
            grid_clone = grid_sample_padding(
                grid_clone, padding_mode, align_corners, borders=[w, h]
            )

        if mode == "bilinear":
            grid_clone += 4
            w_coor = ivy.reshape(grid_clone[..., 0], (n, to_h, to_w))
            h_coor = ivy.reshape(grid_clone[..., 1], (n, to_h, to_w))

            w0 = ivy.astype(ivy.floor(w_coor), ivy.int64)
            h0 = ivy.astype(ivy.floor(h_coor), ivy.int64)
            w1 = w0 + 1
            h1 = h0 + 1

            v00 = ivy.permute_dims(input_clone[batch_coor, :, h0, w0], (0, 3, 1, 2))
            v01 = ivy.permute_dims(input_clone[batch_coor, :, h0, w1], (0, 3, 1, 2))
            v10 = ivy.permute_dims(input_clone[batch_coor, :, h1, w0], (0, 3, 1, 2))
            v11 = ivy.permute_dims(input_clone[batch_coor, :, h1, w1], (0, 3, 1, 2))

            alpha = ivy.reshape(w_coor - w0, (n, 1, to_h, to_w))
            beta = ivy.reshape(h_coor - h0, (n, 1, to_h, to_w))

            alpha = ivy.astype(alpha, ivy.float32)
            beta = ivy.astype(beta, ivy.float32)

            v0 = v00 * (1 - alpha) + v01 * alpha
            v1 = v10 * (1 - alpha) + v11 * alpha

            return v0 * (1 - beta) + v1 * beta

        elif mode == "nearest":
            w_coor = ivy.reshape(grid_clone[..., 0], (n, to_h, to_w))
            h_coor = ivy.reshape(grid_clone[..., 1], (n, to_h, to_w))

            w_coor = ivy.astype(ivy.round(w_coor), ivy.int64) + 4
            h_coor = ivy.astype(ivy.round(h_coor), ivy.int64) + 4
            return ivy.permute_dims(
                input_clone[batch_coor, :, h_coor, w_coor], (0, 3, 1, 2)
            )

        else:
            raise ivy.exceptions.IvyError(f"Not supported mode {mode}")

    elif ivy.get_num_dims(input_clone) == 5:  # sample from 3D images
        n, c, d, h, w = input_clone.shape
        n, to_d, to_h, to_w, gc = grid_clone.shape

        # Un-normalize 3D grid
        if align_corners:  # to range[0, size - 1]
            grid_clone[..., 0] = ((grid_clone[..., 0] + 1) / 2) * (w - 1)
            grid_clone[..., 1] = ((grid_clone[..., 1] + 1) / 2) * (h - 1)
            grid_clone[..., 2] = ((grid_clone[..., 2] + 1) / 2) * (d - 1)
        elif not align_corners:  # to range[0.5, size - 0.5]
            grid_clone[..., 0] = ((grid_clone[..., 0] + 1) * w - 1) / 2
            grid_clone[..., 1] = ((grid_clone[..., 1] + 1) * h - 1) / 2
            grid_clone[..., 2] = ((grid_clone[..., 2] + 1) * d - 1) / 2

        batch_coor = ivy.reshape(ivy.arange(n), (-1, 1))
        batch_coor = ivy.repeat(batch_coor, to_d * to_h * to_w, axis=1)
        batch_coor = ivy.reshape(batch_coor, (n, to_d, to_h, to_w))
        padding = [(0, 0) for _ in range(2)] + [(3, 3) for _ in range(3)]
        input_clone = ivy.pad(input_clone, padding, mode="constant", constant_values=0)

        grid_clone = grid_sample_padding(
            grid_clone, padding_mode, align_corners, borders=[w, h, d]
        )

        if mode == "bilinear":
            grid_clone += 3
            w_coor = ivy.reshape(grid_clone[..., 0], (n, to_d, to_h, to_w))
            h_coor = ivy.reshape(grid_clone[..., 1], (n, to_d, to_h, to_w))
            d_coor = ivy.reshape(grid_clone[..., 2], (n, to_d, to_h, to_w))

            w0 = ivy.astype(ivy.floor(w_coor), ivy.int64)
            h0 = ivy.astype(ivy.floor(h_coor), ivy.int64)
            d0 = ivy.astype(ivy.floor(d_coor), ivy.int64)
            w1 = w0 + 1
            h1 = h0 + 1
            d1 = d0 + 1

            v000 = ivy.permute_dims(
                input_clone[batch_coor, :, d0, h0, w0], (0, 4, 1, 2, 3)
            )  # tnw
            v001 = ivy.permute_dims(
                input_clone[batch_coor, :, d0, h0, w1], (0, 4, 1, 2, 3)
            )  # tne
            v010 = ivy.permute_dims(
                input_clone[batch_coor, :, d0, h1, w0], (0, 4, 1, 2, 3)
            )  # tsw
            v011 = ivy.permute_dims(
                input_clone[batch_coor, :, d0, h1, w1], (0, 4, 1, 2, 3)
            )  # tse
            v100 = ivy.permute_dims(
                input_clone[batch_coor, :, d1, h0, w0], (0, 4, 1, 2, 3)
            )  # bnw
            v101 = ivy.permute_dims(
                input_clone[batch_coor, :, d1, h0, w1], (0, 4, 1, 2, 3)
            )  # bne
            v110 = ivy.permute_dims(
                input_clone[batch_coor, :, d1, h1, w0], (0, 4, 1, 2, 3)
            )  # bsw
            v111 = ivy.permute_dims(
                input_clone[batch_coor, :, d1, h1, w1], (0, 4, 1, 2, 3)
            )  # bse

            alpha = ivy.reshape(w_coor - w0, (n, 1, to_d, to_h, to_w))
            beta = ivy.reshape(h_coor - h0, (n, 1, to_d, to_h, to_w))
            gamma = ivy.reshape(d_coor - d0, (n, 1, to_d, to_h, to_w))

            alpha = ivy.astype(alpha, ivy.float32)
            beta = ivy.astype(beta, ivy.float32)
            gamma = ivy.astype(gamma, ivy.float32)

            v = (alpha * beta * gamma) * v111
            v += ((1 - alpha) * beta * gamma) * v110
            v += (alpha * (1 - beta) * gamma) * v101
            v += ((1 - alpha) * (1 - beta) * gamma) * v100

            v += (alpha * beta * (1 - gamma)) * v011
            v += ((1 - alpha) * beta * (1 - gamma)) * v010
            v += (alpha * (1 - beta) * (1 - gamma)) * v001
            v += ((1 - alpha) * (1 - beta) * (1 - gamma)) * v000
            return v

        elif mode == "nearest":
            ceil_mask = grid_clone % 1 == 0.5
            grid_clone[ceil_mask] = ivy.astype(
                ivy.ceil(grid_clone[ceil_mask]), ivy.int64
            )

            w_coor = ivy.reshape(grid_clone[..., 0], (n, to_d, to_h, to_w))
            h_coor = ivy.reshape(grid_clone[..., 1], (n, to_d, to_h, to_w))
            d_coor = ivy.reshape(grid_clone[..., 2], (n, to_d, to_h, to_w))

            w_coor = ivy.astype(ivy.round(w_coor), ivy.int64) + 3
            h_coor = ivy.astype(ivy.round(h_coor), ivy.int64) + 3
            d_coor = ivy.astype(ivy.round(d_coor), ivy.int64) + 3
            return ivy.permute_dims(
                input_clone[batch_coor, :, d_coor, h_coor, w_coor], (0, 4, 1, 2, 3)
            )

        elif mode == "bicubic":
            raise ivy.exceptions.IvyError("Bicubic is not support in 3D grid sampling")

    else:
        raise ivy.exceptions.IvyError(f"Not supported input shape {input_clone.shape}")


def grid_sample_padding(grid, padding_mode, align_corners, borders=None):
    if padding_mode == "reflection":
        if align_corners:
            for idx, border in enumerate(borders):
                grid[..., idx] = reflect(grid[..., idx], 0, 2 * (border - 1))
                grid[..., idx] = ivy.clip(grid[..., idx], 0, border - 1)

        else:
            for idx, border in enumerate(borders):
                grid[..., idx] = reflect(grid[..., idx], -1, 2 * border - 1)
                grid[..., idx] = ivy.clip(grid[..., idx], 0, border - 1)

    elif padding_mode == "border":
        for idx, border in enumerate(borders):
            grid[..., idx] = ivy.clip(grid[..., idx], 0, border - 1)

    masks = []
    for idx, border in enumerate(borders):
        masks.append(ivy.bitwise_or(grid[..., idx] < -4, grid[..., idx] > border + 2))
        borders[idx] += 1

    zeros_mask = masks[0]
    for i in range(1, len(borders)):
        zeros_mask = ivy.bitwise_or(zeros_mask, masks[i])

    if grid[zeros_mask].shape[0] > 0:
        grid[zeros_mask] = ivy.array(borders)
    return grid


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def interpolate(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias=False,
):
    if (
        mode not in ["linear", "bilinear", "bicubic", "trilinear"]
        and align_corners is not None
    ):
        raise ivy.utils.exceptions.IvyException(
            "align_corners option can only be set with the interpolating"
            f"modes: linear | bilinear | bicubic | trilinear (got {mode})"
        )
    ivy.utils.assertions.check_elem_in_list(
        ivy.get_num_dims(input),
        range(3, 6),
        message=(
            "Input Error: Only 3D, 4D and 5D input Tensors supported (got"
            f" {ivy.get_num_dims(input)}D) for the modes: nearest | linear | bilinear |"
            f" bicubic | trilinear | area | nearest-exact (got {mode})"
        ),
    )
    return ivy.interpolate(
        input,
        size,
        mode=mode,
        scale_factor=scale_factor,
        recompute_scale_factor=recompute_scale_factor,
        align_corners=bool(align_corners),
        antialias=antialias,
    )


@to_ivy_arrays_and_back
def pad(input, pad, mode="constant", value=0):
    # deal with any negative pad values
    if any([pad_value < 0 for pad_value in pad]):
        pad = list(pad)
        slices = []
        for n in reversed(range(len(pad) // 2)):
            i = n * 2
            j = i + 1
            start = None
            stop = None
            if pad[i] < 0:
                start = -pad[i]
                pad[i] = 0
            if pad[j] < 0:
                stop = pad[j]
                pad[j] = 0
            slices.append(slice(start, stop))
        ndim = len(input.shape)
        while len(slices) < ndim:
            slices.insert(0, slice(None))
        input = input[tuple(slices)]

    value = 0 if value is None else value
    mode_dict = {
        "constant": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }
    if mode not in mode_dict:
        raise ValueError(f"Unsupported padding mode: {mode}")
    pad = _handle_padding_shape(pad, len(input.shape), mode)
    return ivy.pad(input, pad, mode=mode_dict[mode], constant_values=value)


@to_ivy_arrays_and_back
def pixel_shuffle(input, upscale_factor):
    input_shape = ivy.shape(input)

    ivy.utils.assertions.check_equal(
        ivy.get_num_dims(input),
        4,
        message=(
            "pixel_shuffle expects 4D input, but got input with sizes"
            f" {str(input_shape)}"
        ),
        as_array=False,
    )
    b = input_shape[0]
    c = input_shape[1]
    h = input_shape[2]
    w = input_shape[3]
    upscale_factor_squared = upscale_factor * upscale_factor
    ivy.utils.assertions.check_equal(
        c % upscale_factor_squared,
        0,
        message="pixel_shuffle expects input channel to be divisible by square "
        + "of upscale_factor, but got input with sizes "
        + str(input_shape)
        + ", upscale_factor="
        + str(upscale_factor)
        + ", and self.size(1)="
        + str(c)
        + " is not divisible by "
        + str(upscale_factor_squared),
        as_array=False,
    )
    oc = int(c / upscale_factor_squared)
    oh = h * upscale_factor
    ow = w * upscale_factor

    input_reshaped = ivy.reshape(input, (b, oc, upscale_factor, upscale_factor, h, w))
    return ivy.reshape(
        ivy.permute_dims(input_reshaped, (0, 1, 4, 2, 5, 3)), (b, oc, oh, ow)
    )


@to_ivy_arrays_and_back
def pixel_unshuffle(input, downscale_factor):
    input_shape = ivy.shape(input)

    ivy.utils.assertions.check_equal(
        ivy.get_num_dims(input),
        4,
        message=(
            f"pixel_unshuffle expects 4D input, but got input with sizes {input_shape}"
        ),
        as_array=False,
    )

    b = input_shape[0]
    c = input_shape[1]
    h = input_shape[2]
    w = input_shape[3]
    downscale_factor_squared = downscale_factor * downscale_factor

    ivy.utils.assertions.check_equal(
        [h % downscale_factor, w % downscale_factor],
        [0, 0],  # Assert h % downscale_factor == 0 and w % downscale_factor == 0
        message=(
            "pixel_unshuffle expects input height and width to be divisible by "
            f"downscale_factor, but got input with sizes {input_shape}"
            f", downscale_factor= {downscale_factor}"
            f", and either self.size(2)= {h}"
            f" or self.size(3)= {w}"
            f" is not divisible by {downscale_factor}"
        ),
        as_array=False,
    )
    oc = c * downscale_factor_squared
    oh = int(h / downscale_factor)
    ow = int(w / downscale_factor)

    input_reshaped = ivy.reshape(
        input, (b, c, oh, downscale_factor, ow, downscale_factor)
    )
    return ivy.reshape(
        ivy.permute_dims(input_reshaped, (0, 1, 3, 5, 2, 4)), (b, oc, oh, ow)
    )


def reflect(x, low2, high2):
    min = low2 / 2
    span = (high2 - low2) / 2
    x = ivy.abs(x - min)
    frac_in = ivy.abs(x / span)
    extra = (frac_in - ivy.floor(frac_in)) * ivy.abs(span)
    flips = ivy.floor(x / span)
    x[flips % 2 == 0] = (extra + min)[flips % 2 == 0]
    x[flips % 2 != 0] = (span - extra + min)[flips % 2 != 0]
    return x


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def upsample(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
):
    return interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def upsample_bilinear(input, size=None, scale_factor=None):
    return interpolate(
        input, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True
    )


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def upsample_nearest(input, size=None, scale_factor=None):
    return interpolate(input, size=size, scale_factor=scale_factor, mode="nearest")
