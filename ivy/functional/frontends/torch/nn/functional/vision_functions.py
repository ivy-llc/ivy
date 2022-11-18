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


def _pad_handle_padding_shape(padding, n, mode):
    if isinstance(padding[0], (list, tuple)):  # case nested
        padding = tuple([tuple(i) for i in ivy.flip(ivy.array(list(padding)), axis=0)])
    elif len(padding) == 1:  # case scalar
        padding = (padding[0], padding[0])
    else:  # case flat
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
        padding = tuple([tuple(i) for i in ivy.flip(ivy.array(list(padding)), axis=0)])
    return padding


@to_ivy_arrays_and_back
def pad(input, pad, mode="constant", value=0):
    pad = _pad_handle_padding_shape(pad, len(input.shape), mode)
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
