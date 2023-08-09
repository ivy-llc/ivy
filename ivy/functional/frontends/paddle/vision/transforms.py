import ivy
from ivy.func_wrapper import with_supported_dtypes
from ..tensor.tensor import Tensor
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def to_tensor(pic, data_format="CHW"):
    array = ivy.array(pic)
    return Tensor(array)


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


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def adjust_hue(img, hue_factor):
    img, hue_factor = ivy.functional.frontends.paddle.promote_types_of_paddle_inputs(
        img, hue_factor
    )
    assert (
        hue_factor >= -0.5 and hue_factor <= 0.5
    ), "hue_factor should be in range [-0.5, 0.5]"

    img_hsv = _rgb_to_hsv(img)
    h, s, v = img_hsv[0], img_hsv[1], img_hsv[2]

    h = h + hue_factor
    h = h - ivy.floor(h)
    img_adjusted = _hsv_to_rgb(ivy.stack([h, s, v], axis=-3))

    return img_adjusted
