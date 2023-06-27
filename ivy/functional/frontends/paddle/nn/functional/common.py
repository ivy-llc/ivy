# local
import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
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


def get_mask(shape, device, prob, seed=None):
    mask = ivy.where(
        ivy.random_uniform(shape=shape, device=device, seed=seed) < prob,
        0.0,
        1.0,
    )
    return mask


@with_supported_dtypes({"2.4.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
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
