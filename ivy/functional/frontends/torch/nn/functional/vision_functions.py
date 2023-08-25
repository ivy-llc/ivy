# global
import math

# local
import ivy
from ivy import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.utils.exceptions import IvyNotImplementedException


@to_ivy_arrays_and_back
def pixel_shuffle(input, upscale_factor):
    input_shape = ivy.shape(input)

    ivy.utils.assertions.check_equal(
        ivy.get_num_dims(input),
        4,
        message="pixel_shuffle expects 4D input, but got input with sizes "
        + str(input_shape),
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
    ),

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


def _handle_padding_shape(padding, n, mode):
    padding = tuple(
        [
            (padding[i * 2], padding[i * 2 + 1])
            for i in range(int(len(padding) / 2) - 1, -1, -1)
        ]
    )
    while len(padding) < n:
        if mode == "circular":
            padding = padding + ((0, 0),)
        else:
            padding = ((0, 0),) + padding
    if mode == "circular":
        padding = tuple(list(padding)[::-1])
    return padding


@to_ivy_arrays_and_back
def pad(input, pad, mode="constant", value=0):
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


def _get_new_width_height(w_old, h_old, size=None, scale_factor=None):
    if scale_factor and (not size):
        if type(scale_factor) == int:
            h_new = int(w_old * scale_factor)
            w_new = int(h_old * scale_factor)
        elif type(scale_factor) == tuple:
            h_new = int(w_old * scale_factor[0])
            w_new = int(h_old * scale_factor[1])
    elif (not scale_factor) and size:
        if type(size) == int:
            h_new = size
            w_new = size
        elif type(size) == tuple:
            h_new, w_new = size
    return h_new, w_new


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
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
    if mode in ["nearest", "area", "nearest-exact"]:
        ivy.utils.assertions.check_exists(
            align_corners,
            inverse=True,
            message=(
                "align_corners option can only be set with the interpolating modes:"
                " linear | bilinear | bicubic | trilinear"
            ),
        )
    else:
        if not ivy.exists(align_corners):
            align_corners = False

    dim = ivy.get_num_dims(input) - 2  # Number of spatial dimensions.

    if ivy.exists(size) and ivy.exists(scale_factor):
        raise ivy.utils.exceptions.IvyException(
            "only one of size or scale_factor should be defined"
        )

    elif ivy.exists(size) and not ivy.exists(scale_factor):
        scale_factors = None

        if isinstance(size, (list, tuple)):
            ivy.utils.assertions.check_equal(
                len(size),
                dim,
                inverse=False,
                message=(
                    "Input and output must have the "
                    "same number of spatial dimensions,"
                    f" but got input with spatial dimensions of {list(input.shape[2:])}"
                    f" and output size of {size}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format"
                    " and output size in (o1, o2, ...,oK) format."
                ),
                as_array=False,
            )
            output_size = size
        else:
            output_size = [size for _ in range(dim)]

    elif ivy.exists(scale_factor) and not ivy.exists(size):
        output_size = None

        if isinstance(scale_factor, (list, tuple)):
            ivy.utils.assertions.check_equal(
                len(scale_factor),
                dim,
                inverse=False,
                message=(
                    "Input and scale_factor must have the "
                    "same number of spatial dimensions,"
                    f" but got input with spatial dimensions of {list(input.shape[2:])}"
                    f" and scale_factor of shape {scale_factor}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format"
                    " and scale_factor in (s1, s2, ...,sK) format."
                ),
                as_array=False,
            )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]

    else:
        ivy.utils.assertions.check_any(
            [ivy.exists(size), ivy.exists(scale_factor)],
            message="either size or scale_factor should be defined",
            as_array=False,
        )

    if (
        ivy.exists(size)
        and ivy.exists(recompute_scale_factor)
        and bool(recompute_scale_factor)
    ):
        raise ivy.utils.exceptions.IvyException(
            "recompute_scale_factor is not meaningful with an explicit size."
        )

    if ivy.exists(scale_factors):
        output_size = [
            math.floor(ivy.shape(input)[i + 2] * scale_factors[i]) for i in range(dim)
        ]

    if (
        bool(antialias)
        and not (mode in ["bilinear", "bicubic"])
        and ivy.get_num_dims(input) == 4
    ):
        raise ivy.utils.exceptions.IvyException(
            "recompute_scale_factor is not meaningful with an explicit size."
        )

    if ivy.get_num_dims(input) == 3 and mode == "bilinear":
        raise IvyNotImplementedException(
            "Got 3D input, but bilinear mode needs 4D input"
        )
    if ivy.get_num_dims(input) == 3 and mode == "trilinear":
        raise IvyNotImplementedException(
            "Got 3D input, but trilinear mode needs 5D input"
        )
    if ivy.get_num_dims(input) == 4 and mode == "linear":
        raise IvyNotImplementedException("Got 4D input, but linear mode needs 3D input")
    if ivy.get_num_dims(input) == 4 and mode == "trilinear":
        raise IvyNotImplementedException(
            "Got 4D input, but trilinear mode needs 5D input"
        )
    if ivy.get_num_dims(input) == 5 and mode == "linear":
        raise IvyNotImplementedException("Got 5D input, but linear mode needs 3D input")
    if ivy.get_num_dims(input) == 5 and mode == "bilinear":
        raise IvyNotImplementedException(
            "Got 5D input, but bilinear mode needs 4D input"
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
        input, output_size, mode=mode, align_corners=align_corners, antialias=antialias
    )


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def upsample_nearest(input, size=None, scale_factor=None):
    return interpolate(input, size=size, scale_factor=scale_factor, mode="nearest")


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def upsample_bilinear(input, size=None, scale_factor=None):
    return interpolate(
        input, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True
    )


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
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


def reflect(x, low2, high2):
    min = low2 / 2
    span = (high2 - low2) / 2
    x = ivy.abs(x - min)
    frac_in = ivy.abs(x / span)
    extra = (frac_in - ivy.floor(frac_in)) * ivy.abs(span)
    flips = ivy.floor(x / span)
    x *= 0
    x[flips % 2 == 0] += (extra + min)[flips % 2 == 0]
    x[flips % 2 != 0] += (span - extra + min)[flips % 2 != 0]
    return x

def bicubic_interp(fx, alpha = -0.75):
    coeffs = []

    fx0 = fx + 1
    fx2 = 1 - fx

    coeffs.append(alpha * fx0 * fx0 * fx0 - (5 * alpha * fx0 * fx0) + (8 * alpha * fx0) - (4 * alpha))
    coeffs.append((alpha + 2) * fx * fx * fx - ((alpha + 3) * fx * fx) + 1)
    coeffs.append((alpha + 2) * fx2 * fx2 * fx2 - ((alpha + 3) * fx2 * fx2) + 1)
    coeffs.append(1 - coeffs[0] - coeffs[1] - coeffs[2])
    return coeffs

border_clamp = lambda grid, border: ivy.fmin(border, ivy.fmax(grid, 0))
# @with_supported_dtypes({"2.0.1 and below": ("float32", "float64")}, "torch")
@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    # Ref:
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/GridSampler.cpp
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/GridSamplerKernel.cpp
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/GridSampler.cu
    # https://github.com/Tencent/ncnn/blob/cb674ac5eddb32f0709a60c81f71d2cbc6bc89da/src/layer/gridsample.cpp#L21
    if ivy.get_num_dims(input) == 4: # sample from 2D images
        n, c, h, w = input.shape
        n, to_h, to_w, gc = grid.shape

        # padding images so we can access padding values by indexing
        # Padding for h and w

        # Un-normalize 2D grid
        if align_corners: # to range[0, size - 1]
            grid[..., 0] = ((grid[..., 0] + 1) / 2) * (w - 1)
            grid[..., 1] = ((grid[..., 1] + 1) / 2) * (h - 1)

        else: # to range[0.5, size - 0.5]
            grid[..., 0] = ((grid[..., 0] + 1) * w - 1) / 2
            grid[..., 1] = ((grid[..., 1] + 1) * h - 1) / 2

        # compute all coordinate depends on padding mode. Apply padding(zeros, reflect, border)
        if padding_mode == 'reflection':
            if align_corners:
                grid[..., 0] = reflect(grid[..., 0], 0, 2*(w - 1))
                grid[..., 1] = reflect(grid[..., 1], 0, 2*(h - 1))
            else:
                grid[..., 0] = reflect(grid[..., 0], -1, 2*w - 1)
                grid[..., 1] = reflect(grid[..., 1], -1, 2*h - 1)

            grid[..., 0] = border_clamp(grid[..., 0], w - 1)
            grid[..., 1] = border_clamp(grid[..., 1], h - 1)

        elif padding_mode == 'border':
            grid[..., 0] = border_clamp(grid[..., 0], w - 1)
            grid[..., 1] = border_clamp(grid[..., 1], h - 1)

        elif padding_mode == 'zeros':
            w_mask = ivy.bitwise_or(grid[..., 0] < 0, grid[..., 0] > w)
            h_mask = ivy.bitwise_or(grid[..., 1] < 0, grid[..., 1] > h)
            zeros_mask = ivy.bitwise_or(w_mask, h_mask)
            grid[zeros_mask] += (w * h)
            grid[..., 0] = border_clamp(grid[..., 0], w)
            grid[..., 1] = border_clamp(grid[..., 1], h)

        # pad and shift for 1
        padding = [(0, 0) for i in range(2)] + [
            (1, 2),
            (1, 2),
        ]
        input = ivy.pad(input, padding, mode="constant", constant_values=0)
        grid += 1

        # Apply sampling by mode
        batch_coor = ivy.reshape(ivy.arange(n), (-1, 1))
        batch_coor = ivy.repeat(batch_coor, to_h * to_w, axis=1)
        batch_coor = ivy.reshape(batch_coor, (n, to_h, to_w))
        w_coor = ivy.reshape(grid[..., 0], (n, to_h, to_w))
        h_coor = ivy.reshape(grid[..., 1], (n, to_h, to_w))

        if mode == 'bilinear':
            w0 = ivy.astype(ivy.floor(w_coor), ivy.int64)
            h0 = ivy.astype(ivy.floor(h_coor), ivy.int64)
            w1 = w0 + 1
            h1 = h0 + 1

            v00 = ivy.permute_dims(input[batch_coor, :, h0, w0], (0, 3, 1, 2))
            v01 = ivy.permute_dims(input[batch_coor, :, h0, w1], (0, 3, 1, 2))
            v10 = ivy.permute_dims(input[batch_coor, :, h1, w0], (0, 3, 1, 2))
            v11 = ivy.permute_dims(input[batch_coor, :, h1, w1], (0, 3, 1, 2))

            alpha = ivy.reshape(w_coor - w0, (n, 1, to_h, to_w))  # We need to add a dimension here to make alpha and beta have a same shape
            beta = ivy.reshape(h_coor - h0, (n, 1, to_h, to_w))

            v0 = v00 * (1 - alpha) + v01 * alpha
            v1 = v10 * (1 - alpha) + v11 * alpha

            return v0 * (1 - beta) + v1 * beta

        elif mode == 'bicubic':
            w1 = ivy.astype(ivy.floor(w_coor), ivy.int64)
            h1 = ivy.astype(ivy.floor(h_coor), ivy.int64)
            w0 = w1 - 1
            h0 = h1 - 1
            w2 = w0 + 1
            h2 = h0 + 1
            w3 = w0 + 2
            h3 = h0 + 2

            v00 = ivy.permute_dims(input[batch_coor, :, h0, w0], (0, 3, 1, 2))
            v01 = ivy.permute_dims(input[batch_coor, :, h0, w1], (0, 3, 1, 2))
            v02 = ivy.permute_dims(input[batch_coor, :, h0, w2], (0, 3, 1, 2))
            v03 = ivy.permute_dims(input[batch_coor, :, h0, w3], (0, 3, 1, 2))
            v10 = ivy.permute_dims(input[batch_coor, :, h1, w0], (0, 3, 1, 2))
            v11 = ivy.permute_dims(input[batch_coor, :, h1, w1], (0, 3, 1, 2))
            v12 = ivy.permute_dims(input[batch_coor, :, h1, w2], (0, 3, 1, 2))
            v13 = ivy.permute_dims(input[batch_coor, :, h1, w3], (0, 3, 1, 2))
            v20 = ivy.permute_dims(input[batch_coor, :, h2, w0], (0, 3, 1, 2))
            v21 = ivy.permute_dims(input[batch_coor, :, h2, w1], (0, 3, 1, 2))
            v22 = ivy.permute_dims(input[batch_coor, :, h2, w2], (0, 3, 1, 2))
            v23 = ivy.permute_dims(input[batch_coor, :, h2, w3], (0, 3, 1, 2))
            v30 = ivy.permute_dims(input[batch_coor, :, h3, w0], (0, 3, 1, 2))
            v31 = ivy.permute_dims(input[batch_coor, :, h3, w1], (0, 3, 1, 2))
            v32 = ivy.permute_dims(input[batch_coor, :, h3, w2], (0, 3, 1, 2))
            v33 = ivy.permute_dims(input[batch_coor, :, h3, w3], (0, 3, 1, 2))

            w_coeffs = bicubic_interp(w_coor - w1)
            h_coeffs = bicubic_interp(h_coor - h1)


            v0 = v00 * w_coeffs[0] + v01 * w_coeffs[1] + v02 * w_coeffs[2] + v03 * w_coeffs[3]
            v1 = v10 * w_coeffs[0] + v11 * w_coeffs[1] + v12 * w_coeffs[2] + v13 * w_coeffs[3]
            v2 = v20 * w_coeffs[0] + v21 * w_coeffs[1] + v22 * w_coeffs[2] + v23 * w_coeffs[3]
            v3 = v30 * w_coeffs[0] + v31 * w_coeffs[1] + v32 * w_coeffs[2] + v33 * w_coeffs[3]

            return v0 * h_coeffs[0] + v1 * h_coeffs[1] + v2 * h_coeffs[2] + v3 * h_coeffs[3]


        elif mode == 'nearest':
            w_coor = ivy.astype(ivy.round(w_coor), ivy.int64)
            h_coor = ivy.astype(ivy.round(h_coor), ivy.int64)
            return ivy.permute_dims(input[batch_coor, :, h_coor, w_coor], (0, 3, 1, 2))


    elif ivy.get_num_dims(input) == 5: # sample from 3D images
        n, c, d, h, w = input.shape
        n, to_d, to_h, to_w, gc = grid.shape

        # Un-normalize 3D grid
        if align_corners:  # to range[0, size - 1]
            grid[:, :, :, 0] = ((grid[:, :, :, 0] + 1) / 2) * (w - 1)
            grid[:, :, :, 1] = ((grid[:, :, :, 1] + 1) / 2) * (h - 1)
            grid[:, :, :, 2] = ((grid[:, :, :, 2] + 1) / 2) * (d - 1)
        else:  # to range[0.5, size - 0.5]
            grid[:, :, :, 0] = ((grid[:, :, :, 0] + 1) * w - 1) / 2
            grid[:, :, :, 1] = ((grid[:, :, :, 1] + 1) * h - 1) / 2
            grid[:, :, :, 2] = ((grid[:, :, :, 2] + 1) * d - 1) / 2

        if padding_mode == 'reflection':
            if align_corners:
                grid[:, :, :, 0] = reflect(grid[:, :, :, 0], 0, 2*(w - 1))
                grid[:, :, :, 1] = reflect(grid[:, :, :, 1], 0, 2*(h - 1))
                grid[:, :, :, 2] = reflect(grid[:, :, :, 2], 0, 2*(d - 1))
            else:
                grid[:, :, :, 0] = reflect(grid[:, :, :, 0], -1, 2*w - 1)
                grid[:, :, :, 1] = reflect(grid[:, :, :, 1], -1, 2*h - 1)
                grid[:, :, :, 2] = reflect(grid[:, :, :, 2], -1, 2*d - 1)

            grid[..., 0] = border_clamp(grid[..., 0], w - 1)
            grid[..., 1] = border_clamp(grid[..., 1], h - 1)
            grid[..., 2] = border_clamp(grid[..., 2], d - 1)

        elif padding_mode == 'border':
            grid[..., 0] = border_clamp(grid[..., 0], w - 1)
            grid[..., 1] = border_clamp(grid[..., 1], h - 1)
            grid[..., 2] = border_clamp(grid[..., 2], d - 1)


        elif padding_mode == 'zeros':
            w_mask = ivy.bitwise_or(grid[:, :, :, 0] < 0, grid[:, :, :, 0] > w)
            h_mask = ivy.bitwise_or(grid[:, :, :, 1] < 0, grid[:, :, :, 1] > h)
            d_mask = ivy.bitwise_or(grid[:, :, :, 2] < 0, grid[:, :, :, 2] > d)
            zeros_mask = ivy.bitwise_or(w_mask, h_mask)
            zeros_mask = ivy.bitwise_or(zeros_mask, d_mask)

            grid[zeros_mask] += (w * h * d)
            grid[..., 0] = border_clamp(grid[..., 0], w)
            grid[..., 1] = border_clamp(grid[..., 1], h)
            grid[..., 2] = border_clamp(grid[..., 2], d)

        # Padding for d, h, and w
        padding = [(0, 0) for _ in range(2)] + [(0, 2) for _ in range(3)]
        input = ivy.pad(input, padding, mode="constant", constant_values=0)

        batch_coor = ivy.reshape(ivy.arange(n), (-1, 1))
        batch_coor = ivy.repeat(batch_coor, to_d * to_h * to_w , axis=1)
        batch_coor = ivy.reshape(batch_coor, (n, to_d, to_h, to_w))
        w_coor = ivy.reshape(grid[..., 0], (n, to_d, to_h, to_w))
        h_coor = ivy.reshape(grid[..., 1], (n, to_d, to_h, to_w))
        d_coor = ivy.reshape(grid[..., 2], (n, to_d, to_h, to_w))

        if mode == 'bilinear':
            w0 = ivy.astype(ivy.floor(w_coor), ivy.int64)
            h0 = ivy.astype(ivy.floor(h_coor), ivy.int64)
            d0 = ivy.astype(ivy.floor(d_coor), ivy.int64)
            w1 = w0 + 1
            h1 = h0 + 1
            d1 = d0 + 1

            v000 = ivy.permute_dims(input[batch_coor, :, d0, h0, w0], (0, 4, 1, 2, 3))
            v001 = ivy.permute_dims(input[batch_coor, :, d0, h0, w1], (0, 4, 1, 2, 3))
            v010 = ivy.permute_dims(input[batch_coor, :, d0, h1, w0], (0, 4, 1, 2, 3))
            v011 = ivy.permute_dims(input[batch_coor, :, d0, h1, w1], (0, 4, 1, 2, 3))
            v100 = ivy.permute_dims(input[batch_coor, :, d1, h0, w0], (0, 4, 1, 2, 3))
            v101 = ivy.permute_dims(input[batch_coor, :, d1, h0, w1], (0, 4, 1, 2, 3))
            v110 = ivy.permute_dims(input[batch_coor, :, d1, h1, w0], (0, 4, 1, 2, 3))
            v111 = ivy.permute_dims(input[batch_coor, :, d1, h1, w1], (0, 4, 1, 2, 3))

            alpha = ivy.reshape(w_coor - w0, (n, 1, to_d, to_h, to_w))
            beta = ivy.reshape(h_coor - h0, (n, 1, to_d, to_h, to_w))
            gamma = ivy.reshape(d_coor - d0, (n, 1, to_d, to_h, to_w))

            v00 = v000 * (1 - alpha) + v001 * alpha
            v01 = v010 * (1 - alpha) + v011 * alpha
            v10 = v100 * (1 - alpha) + v101 * alpha
            v11 = v110 * (1 - alpha) + v111 * alpha

            v0 = v00 * (1 - beta) + v01 * beta
            v1 = v10 + (1 - beta) + v11 * beta

            return v0 * (1 - gamma) + v1 * gamma
            # tnw = ivy.astype(ivy.floor(grid), ivy.int64)
            # bse = tnw + 1

            # w_tnw = ivy.reshape(tnw[..., 0], (n, to_h, to_w, to_d)) # also the w of tsw, bsw, bnw
            # h_tnw = ivy.reshape(tnw[..., 1], (n, to_h, to_w, to_d)) # also the h of tne, bnw, bne
            # d_tnw = ivy.reshape(tnw[..., 2], (n, to_h, to_w, to_d)) # also the d of all top (tne, tse, tsw)
            #
            # w_bse = ivy.reshape(bse[..., 0], (n, to_h, to_w, to_d)) # the w of tse, tne, bne
            # h_bse = ivy.reshape(bse[..., 1], (n, to_h, to_w, to_d)) # the h of tse, tsw, bsw
            # d_bse = ivy.reshape(bse[..., 2], (n, to_h, to_w, to_d)) # the d of all bottom (bsw, bnw, bne)
        elif mode == 'nearest':
            w_coor = ivy.astype(ivy.round(w_coor), ivy.int64)
            h_coor = ivy.astype(ivy.round(h_coor), ivy.int64)
            d_coor = ivy.astype(ivy.round(d_coor), ivy.int64)
            print(w_coor.shape, h_coor.shape, d_coor.shape, input[batch_coor, :, d_coor, h_coor, w_coor].shape)
            return ivy.permute_dims(input[batch_coor, :, d_coor, h_coor, w_coor], (0, 4, 1, 2, 3))
            # return ivy.permute_dims(input[batch_coor, :, d_coor, h_coor, w_coor], (0, 4, 1, 2, 3))

        elif mode == 'bicubic':
            print("Bicubic is not support in 3D grid sampling")

    else:
        print("Not supported input shape")


