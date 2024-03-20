# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh


tanh = paddle_tanh


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def celu(
    x,
    /,
    *,
    alpha=1.0,
    name=None,
):
    return ivy.celu(x, alpha=alpha)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def elu(
    x,
    /,
    *,
    alpha=1.0,
    name=None,
):
    return ivy.elu(x, alpha=alpha)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def gelu(x, approximate=False, name=None):
    return ivy.gelu(x, approximate=approximate)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def glu(x, axis=-1, name=None):
    size = x.shape[axis]
    ivy.utils.assertions.check_equal(
        size % 2, 0, message="axis size must be divisible by 2", as_array=False
    )
    a, b = ivy.split(x, num_or_size_splits=2, axis=axis)
    return ivy.multiply(a, ivy.sigmoid(b))


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def gumbel_softmax(x, temperature=1.0, hard=False, axis=-1, name=None):
    gumbel_noice = -ivy.log(-ivy.log(ivy.random_uniform(ivy.shape(x) + 1e-20) + 1e-20))
    gumbel_logits = (x + gumbel_noice) / temperature
    y_soft = ivy.softmax(gumbel_logits, axis=axis)

    if hard:
        y_hard = ivy.one_hot(ivy.argmax(y_soft, axis=axis), ivy.shape(y_soft)[axis])
        return y_hard
    else:
        return y_soft


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def hardshrink(x, threshold=0.5, name=None):
    mask = ivy.logical_or(ivy.greater(x, threshold), ivy.less(x, -threshold))
    return ivy.where(mask, x, 0.0)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def hardsigmoid(x, slope=0.1666667, offset=0.5, name=None):
    ret = ivy.minimum(ivy.maximum(ivy.add(ivy.multiply(x, slope), offset), 0), 1)
    return ret


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def hardswish(x, name=None):
    relu6_val = ivy.relu6(ivy.add(x, 3))
    ret = ivy.multiply(x, ivy.divide(relu6_val, 6))
    return ret


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def hardtanh(
    x,
    /,
    *,
    min=-1.0,
    max=1.0,
    name=None,
):
    less = ivy.where(ivy.less(x, min), min, x)
    ret = ivy.where(ivy.greater(x, max), max, less).astype(x.dtype)
    return ret


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def leaky_relu(x, negative_slope=0.01, name=None):
    return ivy.leaky_relu(x)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def log_sigmoid(x, name=None):
    return -ivy.softplus(-x)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def log_softmax(x, axis=-1, dtype=None, name=None):
    x = ivy.astype(x, dtype) if dtype else x
    ret = ivy.log_softmax(x, axis=axis)
    ret = ivy.astype(ret, dtype) if dtype else ret
    return ret


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def mish(x, name=None):
    return ivy.mish(x)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def prelu(x, weight, data_format="NCHW", name=None):
    return ivy.add(ivy.maximum(0, x), ivy.multiply(weight, ivy.minimum(0, x)))


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def relu(x, name=None):
    return ivy.relu(x)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def relu6(x, name=None):
    return ivy.relu6(x)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def relu_(x, name=None):
    ret = ivy.relu(x)
    ivy.inplace_update(x, ret)
    return x


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def rrelu(
    x,
    /,
    *,
    lower=0.125,
    upper=0.3333333333333333,
    training=False,
    name=None,
):
    if lower < 0 or lower > 1:
        raise ValueError(
            "The lower value must be no less than zero or greater than one. Received:"
            f" {lower}."
        )

    if upper < lower:
        raise ValueError(
            "The upper value must be greater than lower value. Received: lower"
            f" {lower}, upper {upper}."
        )

    if upper > 1:
        raise ValueError(
            f"The upper value must be no greater than one. Received: {upper}."
        )

    is_test = not training
    if is_test:
        add = lower + upper
        ret = add * x * 0.5
        out = ivy.where(x >= 0, x, ret)
        return out.astype(x.dtype)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def selu(
    x,
    /,
    *,
    alpha=1.6732632423543772848170429916717,
    scale=1.0507009873554804934193349852946,
    name=None,
):
    if scale <= 1.0:
        raise ValueError(f"The scale must be greater than 1.0. Received: {scale}.")

    if alpha < 0:
        raise ValueError(f"The alpha must be no less than zero. Received: {alpha}.")

    ret = ivy.where(x > 0, x, alpha * ivy.expm1(x))
    arr = scale * ret
    return ivy.astype(arr, x.dtype)


def silu(x, name=None):
    return ivy.silu(x)


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def softmax_(x, axis=-1, dtype=None, name=None):
    ret = ivy.softmax(x, axis=axis)
    ivy.inplace_update(x, ret)
    return x


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def softplus(x, beta=1, threshold=20, name=None):
    return ivy.softplus(x, beta=beta, threshold=threshold)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def softshrink(
    x,
    /,
    *,
    threshold=0.5,
    name=None,
):
    low = ivy.where(ivy.less(x, -threshold), ivy.add(x, threshold), 0)
    up = ivy.where(ivy.greater(x, threshold), ivy.subtract(x, threshold), 0)
    add = ivy.add(low, up)
    return ivy.astype(add, x.dtype)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def softsign(
    x,
    /,
    *,
    name=None,
):
    return ivy.divide(x, ivy.add(1, ivy.abs(x)))


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def swish(x, name=None):
    return ivy.multiply(x, ivy.sigmoid(x))


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def tanh_(x, name=None):
    ret = ivy.tanh(x)
    ivy.inplace_update(x, ret)
    return x
    # else:
    # ToDo implement a correctly after fixing ivy.random_uniform
    # a = ivy.random_normal(low=lower, high=upper)
    # ret = ivy.where(x >= 0, x, ivy.multiply(a, x))
    # return ret.astype(x.dtype)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def tanhshrink(
    x,
    /,
    *,
    name=None,
):
    return ivy.subtract(x, ivy.tanh(x))
