import ivy
from ivy.func_wrapper import (
    with_supported_dtypes,
    with_unsupported_device_and_dtypes,
)
from ..tensor.tensor import Tensor
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


# --- Helpers --- #
# --------------- #


def _blend_images(img1, img2, ratio):
    # TODO: ivy.check_float(img1) returns False for ivy array
    # TODO: when lerp supports int type and when the above issue is fixed,
    # replace this with ivy.check_float(img1)
    max_value = (
        1.0 if ivy.dtype(img1) == "float32" or ivy.dtype(img1) == "float64" else 255.0
    )
    return ivy.astype(
        ivy.lerp(img2, img1, float(ratio)).clip(0, max_value), ivy.dtype(img1)
    )


# helpers
def _get_image_c_axis(data_format):
    if data_format.lower() == "chw":
        return -3
    elif data_format.lower() == "hwc":
        return -1


def _get_image_num_channels(img, data_format):
    return ivy.shape(img)[_get_image_c_axis(data_format)]


def _hsv_to_rgb(img):
    h, s, v = img[0], img[1], img[2]
    f = h * 6.0
    i = ivy.floor(f)
    f = f - i
    i = ivy.astype(i, ivy.int32) % 6

    p = ivy.clip(v * (1.0 - s), 0.0, 1.0)
    q = ivy.clip(v * (1.0 - s * f), 0.0, 1.0)
    t = ivy.clip(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)

    mask = ivy.astype(
        ivy.equal(
            ivy.expand_dims(i, axis=-3),
            ivy.reshape(ivy.arange(6, dtype=ivy.dtype(i)), (-1, 1, 1)),
        ),
        ivy.dtype(img),
    )
    matrix = ivy.stack(
        [
            ivy.stack([v, q, p, p, t, v], axis=-3),
            ivy.stack([t, v, v, q, p, p], axis=-3),
            ivy.stack([p, p, t, v, v, q], axis=-3),
        ],
        axis=-4,
    )
    return ivy.einsum("...ijk, ...xijk -> ...xjk", mask, matrix)


def _rgb_to_hsv(img):
    maxc = ivy.max(img, axis=-3)
    minc = ivy.min(img, axis=-3)

    is_equal = ivy.equal(maxc, minc)
    one_divisor = ivy.ones_like(maxc)
    c_delta = maxc - minc
    s = c_delta / ivy.where(is_equal, one_divisor, maxc)

    r, g, b = img[0], img[1], img[2]
    c_delta_divisor = ivy.where(is_equal, one_divisor, c_delta)

    rc = (maxc - r) / c_delta_divisor
    gc = (maxc - g) / c_delta_divisor
    bc = (maxc - b) / c_delta_divisor

    hr = ivy.where((maxc == r), bc - gc, ivy.zeros_like(maxc))
    hg = ivy.where(
        ((maxc == g) & (maxc != r)),
        rc - bc + 2.0,
        ivy.zeros_like(maxc),
    )
    hb = ivy.where(
        ((maxc != r) & (maxc != g)),
        gc - rc + 4.0,
        ivy.zeros_like(maxc),
    )

    h = (hr + hg + hb) / 6.0 + 1.0
    h = h - ivy.trunc(h)

    return ivy.stack([h, s, maxc], axis=-3)


# --- Main --- #
# ------------ #


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def adjust_brightness(img, brightness_factor):
    assert brightness_factor >= 0, "brightness_factor should be non-negative."
    assert _get_image_num_channels(img, "CHW") in [
        1,
        3,
    ], "channels of input should be either 1 or 3."

    extreme_target = ivy.zeros_like(img)
    return _blend_images(img, extreme_target, brightness_factor)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64", "uint8")}, "paddle")
@to_ivy_arrays_and_back
def adjust_hue(img, hue_factor):
    assert -0.5 <= hue_factor <= 0.5, "hue_factor should be in range [-0.5, 0.5]"

    channels = _get_image_num_channels(img, "CHW")

    if channels == 1:
        return img
    elif channels == 3:
        if ivy.dtype(img) == "uint8":
            img = ivy.astype(img, "float32") / 255.0

        img_hsv = _rgb_to_hsv(img)
        h, s, v = img_hsv[0], img_hsv[1], img_hsv[2]

        h = h + hue_factor
        h = h - ivy.floor(h)

        img_adjusted = _hsv_to_rgb(ivy.stack([h, s, v], axis=-3))

    else:
        raise ValueError("channels of input should be either 1 or 3.")

    return img_adjusted


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def hflip(img):
    img = ivy.array(img)
    return ivy.flip(img, axis=-1)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
def normalize(img, mean, std, data_format="CHW", to_rgb=False):
    if ivy.is_array(img):
        if data_format == "HWC":
            permuted_axes = [2, 0, 1]
        else:
            permuted_axes = [0, 1, 2]

        img_np = ivy.permute(img, permuted_axes)
        normalized_img = ivy.divide(ivy.subtract(img_np, mean), std)
        return normalized_img
    else:
        raise ValueError("Unsupported input format")


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def pad(img, padding, fill=0, padding_mode="constant"):
    dim_size = img.ndim
    if not hasattr(padding, "__len__"):
        if dim_size == 2:
            trans_padding = ((padding, padding), (padding, padding))
        elif dim_size == 3:
            trans_padding = ((0, 0), (padding, padding), (padding, padding))
    elif len(padding) == 2:
        if dim_size == 2:
            trans_padding = ((padding[1], padding[1]), (padding[0], padding[0]))
        elif dim_size == 3:
            trans_padding = ((0, 0), (padding[1], padding[1]), (padding[0], padding[0]))
    elif len(padding) == 4:
        if dim_size == 2:
            trans_padding = ((padding[1], padding[3]), (padding[0], padding[2]))
        elif dim_size == 3:
            trans_padding = ((0, 0), (padding[1], padding[3]), (padding[0], padding[2]))
    else:
        raise ValueError("padding can only be 1D with size 1, 2, 4 only")

    if padding_mode in ["constant", "edge", "reflect", "symmetric"]:
        return ivy.pad(img, trans_padding, mode=padding_mode, constant_values=fill)
    else:
        raise ValueError("Unsupported padding_mode")


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def to_tensor(pic, data_format="CHW"):
    array = ivy.array(pic)
    return Tensor(array)


@with_unsupported_device_and_dtypes(
    {
        "2.6.0 and below": {
            "cpu": ("int8", "uint8", "int16", "float16", "bfloat16", "bool")
        }
    },
    "paddle",
)
@to_ivy_arrays_and_back
def vflip(img, data_format="CHW"):
    if data_format.lower() == "chw":
        axis = -2
    elif data_format.lower() == "hwc":
        axis = -3
    return ivy.flip(img, axis=axis)
