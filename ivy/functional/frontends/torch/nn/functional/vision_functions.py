import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


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
