# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def cosine_similarity(x1, x2, *, axis=1, eps=1e-08):
    if len(x1.shape) == len(x2.shape) and len(x2.shape) >= 2:
        numerator = ivy.sum(x1 * x2, axis=axis)
        x1_squared_norm = ivy.sum(ivy.square(x1), axis=axis)
        x2_squared_norm = ivy.sum(ivy.square(x2), axis=axis)
    else:
        numerator = ivy.sum(x1 * x2)
        x1_squared_norm = ivy.sum(ivy.square(x1))
        x2_squared_norm = ivy.sum(ivy.square(x2))

    x1_norm = ivy.sqrt(x1_squared_norm)
    x2_norm = ivy.sqrt(x2_squared_norm)
    norm_mm = x1_norm * x2_norm
    denominator = ivy.maximum(norm_mm, eps)

    cosine = numerator / denominator
    return cosine


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def dropout2d(x, *, p=0.5, training=True, data_format="NCHW", name=None):
    return ivy.dropout2d(x, p=p, training=training, data_format=data_format)


def get_mask(shape, device, prob, seed=None):
    mask = ivy.where(
        ivy.random_uniform(shape=shape, device=device, seed=seed) < prob,
        0.0,
        1.0,
    )
    return mask


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def dropout(x, p=0.5, axis=None, training=True, mode="upscale_in_train", name=None):
    if axis > 1:
        raise ValueError("Axis value can only be 0 or 1 or None.")
    elif axis is None or (isinstance(axis, list) and len(axis) == 2):
        mask = get_mask(shape=x.shape, device=ivy.dev(x), prob=p, seed=None)
    elif axis == 0:
        mask = get_mask(shape=(x.shape[0], 1), device=ivy.dev(x), prob=p)
        mask = ivy.broadcast_to(mask, x.shape)
    elif axis == 1:
        mask = get_mask(shape=(1, x.shape[1]), device=ivy.dev(x), prob=p)
        mask = ivy.broadcast_to(mask, x.shape)
    if mode == "upscale_in_train":
        if training:
            out = ivy.multiply(x, mask)
            ret = ivy.multiply(out, 1.0 / (1.0 - p))
        else:
            ret = x
    else:
        if training:
            ret = ivy.multiply(x, mask)
        else:
            ret = ivy.multiply(x, (1.0 - p))
    return ret


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def zeropad2d(x, padding, data_format="NCHW", name=None):
    if ivy.is_array(padding):
        padding = padding.to_list()
    if isinstance(padding, int):
        padding = [padding, padding, padding, padding]
    if len(padding) != 4:
        raise ValueError("Padding length should be 4.")
    if x.ndim != 4:
        raise ValueError("Input x must be 4-dimensional.")
    if data_format == "NCHW":
        padding = ((0, 0), (0, 0), (padding[2], padding[3]), (padding[0], padding[1]))
    elif data_format == "NHWC":
        padding = ((0, 0), (padding[2], padding[3]), (padding[0], padding[1]), (0, 0))
    else:
        raise ValueError("Unknown data_format: {}".format(data_format))
    return ivy.pad(x, padding, mode="constant", constant_values=0.0)


@to_ivy_arrays_and_back
def pad(x, pad, mode="constant", value=0.0, name=None):
    pad = pad.to_list() if ivy.is_array(pad) else pad
    return ivy.pad(x, pad, mode=mode.lower(), value=value)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def interpolate(
    x,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=False,
    align_mode=0,
    data_format="NCHW",
    name=None,
):
    return ivy.interpolate(
        x, size, mode=mode, scale_factor=scale_factor, align_corners=align_corners
    )


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def linear(x, weight, bias=None, name=None):
    weight = ivy.swapaxes(weight, -1, -2)
    return ivy.linear(x, weight, bias=bias)
