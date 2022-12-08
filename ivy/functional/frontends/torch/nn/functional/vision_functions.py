from typing import List, Optional
import ivy
from ivy.exceptions import IvyNotImplementedException
from ivy.functional.backends.torch.experimental.layers import avg_pool1d, avg_pool2d, avg_pool3d
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes

from ivy.functional.frontends.torch.tensor import Tensor


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def interpolate(input: Tensor, size: Optional[int] = None, scale_factor: Optional[List[float]] = None, 
                mode: str = 'nearest', align_corners: Optional[bool] = None, 
                recompute_scale_factor: Optional[bool] = None, antialias: bool = False) -> Tensor:

    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        if align_corners is None:
            align_corners = False

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "Input and output must have the same number of spatial dimensions, but got "
                    f"input with spatial dimensions of {list(input.shape[2:])} and output size of {size}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "output size in (o1, o2, ...,oK) format."

                )
    elif scale_factor is not None:
        assert size is None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "Input and scale_factor must have the same number of spatial dimensions, but "
                    f"got input with spatial dimensions of {list(input.shape[2:])} and "
                    f"scale_factor of shape {scale_factor}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "scale_factor in (s1, s2, ...,sK) format."
                )

    else:
        raise ValueError("either size or scale_factor should be defined")

    if recompute_scale_factor is not None and recompute_scale_factor and size is not None:
        raise ValueError(
            "recompute_scale_factor is not meaningful with an explicit size.")

    if input.dim() == 3 and mode == "bilinear":
        raise IvyNotImplementedException(
            "Got 3D input, but bilinear mode needs 4D input")
    if input.dim() == 3 and mode == "trilinear":
        raise IvyNotImplementedException(
            "Got 3D input, but trilinear mode needs 5D input")
    if input.dim() == 4 and mode == "linear":
        raise IvyNotImplementedException("Got 4D input, but linear mode needs 3D input")
    if input.dim() == 4 and mode == "trilinear":
        raise IvyNotImplementedException(
            "Got 4D input, but trilinear mode needs 5D input")
    if input.dim() == 5 and mode == "linear":
        raise IvyNotImplementedException("Got 5D input, but linear mode needs 3D input")
    if input.dim() == 5 and mode == "bilinear":
        raise IvyNotImplementedException(
            "Got 5D input, but bilinear mode needs 4D input")

    return ivy.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners,
                           recompute_scale_factor=recompute_scale_factor, antialias=antialias)


@to_ivy_arrays_and_back
def pixel_shuffle(input, upscale_factor):

    input_shape = ivy.shape(input)

    ivy.assertions.check_equal(
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
    ivy.assertions.check_equal(
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

    ivy.assertions.check_equal(
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

    ivy.assertions.check_equal(
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
        raise ivy.exceptions.IvyException(
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


@to_ivy_arrays_and_back
def upsample_bilinear(input, size=None, scale_factor=None):
    if scale_factor and size:
        raise ivy.exceptions.IvyException(
            ("only one of size or scale_factor should be defined")
        )
    elif (not scale_factor) and (not size):
        raise ivy.exceptions.IvyException(
            ("either size or scale_factor should be defined")
        )
    elif ivy.get_num_dims(input) != 4:
        raise ivy.exceptions.IvyException(
            (
                f"Got {ivy.get_num_dims(input)}D input,",
                "but bilinear mode needs 4D input (n,c,h,w)",
            )
        )

    n, c, h_old, w_old = input.shape
    h_new, w_new = _get_new_width_height(w_old, h_old, size, scale_factor)

    x_distances_old = ivy.linspace(0, 1, w_old)
    y_distances_old = ivy.linspace(0, 1, h_old)
    x_distances_new = ivy.linspace(0, 1, w_new)
    y_distances_new = ivy.linspace(0, 1, h_new)

    normalize_distances_coeff_x = 1 / (w_old - 1) if w_old != 1 else 1
    normalize_distances_coeff_y = 1 / (h_old - 1) if h_old != 1 else 1

    lower_pivots_x = ivy.zeros(w_new, dtype=ivy.int64)
    higher_pivots_x = ivy.zeros(w_new, dtype=ivy.int64)
    lower_pivots_y = ivy.zeros(h_new, dtype=ivy.int64)
    higher_pivots_y = ivy.zeros(h_new, dtype=ivy.int64)

    for i in range(w_old):
        lower_pivots_x = ivy.where(
            x_distances_old[i] <= x_distances_new, i, lower_pivots_x
        )
    for i in range(h_old):
        lower_pivots_y = ivy.where(
            y_distances_old[i] <= y_distances_new, i, lower_pivots_y
        )
    for i in range(w_old - 1, -1, -1):
        higher_pivots_x = ivy.where(
            x_distances_old[i] >= x_distances_new, i, higher_pivots_x
        )
    for i in range(h_old - 1, -1, -1):
        higher_pivots_y = ivy.where(
            y_distances_old[i] >= y_distances_new, i, higher_pivots_y
        )

    temp = ivy.zeros((n, c, h_old, w_new))
    temp = ivy.where(
        (lower_pivots_x == higher_pivots_x)[None, None, None, :],
        ivy.gather(input, lower_pivots_x, axis=3),
        temp,
    )
    temp = ivy.where(
        (lower_pivots_x != higher_pivots_x)[None, None, None, :],
        ivy.divide(
            ivy.add(
                ivy.multiply(
                    (ivy.gather(x_distances_old, higher_pivots_x) - x_distances_new)[
                        None, None, None, :
                    ],
                    ivy.gather(input, lower_pivots_x, axis=3),
                ),
                ivy.multiply(
                    (x_distances_new - ivy.gather(x_distances_old, lower_pivots_x))[
                        None, None, None, :
                    ],
                    ivy.gather(input, higher_pivots_x, axis=3),
                ),
            ),
            normalize_distances_coeff_x,
        ),
        temp,
    )
    result = ivy.zeros((n, c, h_new, w_new))
    result = ivy.where(
        (lower_pivots_y == higher_pivots_y)[None, None, :, None],
        ivy.gather(temp, lower_pivots_y, axis=2),
        result,
    )
    result = ivy.where(
        (lower_pivots_y != higher_pivots_y)[None, None, :, None],
        ivy.divide(
            ivy.add(
                ivy.multiply(
                    (ivy.gather(y_distances_old, higher_pivots_y) - y_distances_new)[
                        None, None, :, None
                    ],
                    ivy.gather(temp, lower_pivots_y, axis=2),
                ),
                ivy.multiply(
                    (y_distances_new - ivy.gather(y_distances_old, lower_pivots_y))[
                        None, None, :, None
                    ],
                    ivy.gather(temp, higher_pivots_y, axis=2),
                ),
            ),
            normalize_distances_coeff_y,
        ),
        result,
    )

    return result
