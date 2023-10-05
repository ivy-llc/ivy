# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes

from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


# ToDo: this function will be simplified once ivy.alpha_dropout is implemented
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("uint16", "uint32", "uint64")}, "torch")
def alpha_dropout(input, p=0.5, training=False, inplace=False):
    if p == 0.0 or not training or input.shape == () or input.shape == (0,):
        return input
    alpha = 1.7580993408473766
    a = float(1.0 / ivy.sqrt((alpha * alpha * p + 1) * (1 - p)))
    mask = ivy.where(
        ivy.random_uniform(shape=input.shape, device=ivy.dev(input)) < p,
        0.0,
        1.0,
    )
    b = ((mask - 1) * alpha * a) + alpha * a * p
    mask *= a

    if inplace:
        ivy.inplace_update(input, mask * input + b)
        return input
    else:
        masked = mask * input + b
        return masked


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def dropout(input, p=0.5, training=True, inplace=False):
    return ivy.dropout(input, p, training=training)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def dropout1d(input, p=0.5, training=True, inplace=False):
    if inplace:
        return ivy.dropout1d(input, p, training=training, data_format="NCW", out=input)
    return ivy.dropout1d(input, p, training=training, data_format="NCW")


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def dropout2d(input, p=0.5, training=True, inplace=False):
    if input.ndim < 2:
        raise ValueError("Feature dropout requires at least 2 dimensions in the input")

    ret = ivy.dropout2d(input, p, training=training, data_format="NCHW")
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
def dropout3d(input, p=0.5, training=True, inplace=False):
    if inplace:
        return ivy.dropout3d(
            input, p, training=training, data_format="NDHWC", out=input
        )
    return ivy.dropout3d(input, p, training=training, data_format="NDHWC")


# ToDo: this function will be simplified once ivy.feature_alpha_dropout is implemented
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("uint16", "uint32", "uint64")}, "torch")
def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    if p == 0.0 or not training or len(input.shape) < 4:
        return input
    alpha = 1.7580993408473766
    a = float(1.0 / ivy.sqrt((alpha * alpha * p + 1) * (1 - p)))

    mask_shape = input.shape[0:2]
    for _ in range(2, len(input.shape)):
        mask_shape += (1,)
    feature_mask = ivy.where(
        ivy.random_uniform(shape=mask_shape, device=ivy.dev(input)) < p,
        0.0,
        1.0,
    )
    mask = ivy.ones(input.shape) * feature_mask

    b = ((mask - 1) * alpha * a) + alpha * a * p
    mask *= a

    if inplace:
        ivy.inplace_update(input, mask * input + b)
        return input
    else:
        masked = mask * input + b
        return masked
