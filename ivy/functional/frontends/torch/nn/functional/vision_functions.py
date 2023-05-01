# global
import math

# local
import ivy
from ivy import with_unsupported_dtypes
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
            f"pixel_unshuffle expects 4D input, "
            f"but got input with sizes {input_shape}"
        ),
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
            f"pixel_unshuffle expects input height and width to be divisible by "
            f"downscale_factor, but got input with sizes {input_shape}"
            f", downscale_factor= {downscale_factor}"
            f", and either self.size(2)= {h}"
            f" or self.size(3)= {w}"
            f" is not divisible by {downscale_factor}"
        ),
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
    pad = _handle_padding_shape(pad, len(input.shape), mode)
    if mode == "constant":
        return ivy.pad(input, pad, mode="constant", constant_values=value)
    elif mode == "reflect":
        return ivy.pad(input, pad, mode="reflect", reflect_type="even")
    elif mode == "replicate":
        return ivy.pad(input, pad, mode="edge")
    elif mode == "circular":
        return ivy.pad(input, pad, mode="wrap")
    else:
        raise ivy.utils.exceptions.IvyException(
            (
                "mode '{}' must be in "
                + "['constant', 'reflect', 'replicate', 'circular']"
            ).format(mode)
        )


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
        "1.11.0 and below": (
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
            message="align_corners option can only be set with the interpolating modes:"
            " linear | bilinear | bicubic | trilinear",
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
                message=f"Input and output must have the "
                f"same number of spatial dimensions,"
                f" but got input with spatial dimensions of {list(input.shape[2:])}"
                f" and output size of {size}. "
                f"Please provide input tensor in (N, C, d1, d2, ...,dK) format"
                f" and output size in (o1, o2, ...,oK) format.",
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
                message=f"Input and scale_factor must have the "
                f"same number of spatial dimensions,"
                f" but got input with spatial dimensions of {list(input.shape[2:])}"
                f" and scale_factor of shape {scale_factor}. "
                f"Please provide input tensor in (N, C, d1, d2, ...,dK) format"
                f" and scale_factor in (s1, s2, ...,sK) format.",
            )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]

    else:
        ivy.utils.assertions.check_any(
            [ivy.exists(size), ivy.exists(scale_factor)],
            message="either size or scale_factor should be defined",
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
        message=f"Input Error: Only 3D, 4D and 5D input Tensors supported "
        f"(got {ivy.get_num_dims(input)}D) for the modes: nearest | linear | bilinear "
        f"| bicubic | trilinear | area | nearest-exact (got {mode})",
    )

    return ivy.interpolate(
        input, output_size, mode=mode, align_corners=align_corners, antialias=antialias
    )


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
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


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def upsample_nearest(input, size=None, scale_factor=None):
    return interpolate(input, size=size, scale_factor=scale_factor, mode="nearest")


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def upsample_bilinear(input, size=None, scale_factor=None):
    return interpolate(
        input, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True
    )
