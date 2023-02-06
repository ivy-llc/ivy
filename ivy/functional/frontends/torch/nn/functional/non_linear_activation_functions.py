# global
from itertools import product
import math

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


def _compute_threshold(input, threshold, value, inplace):
    ret = ivy.where(ivy.greater(input, threshold), input, value)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def _compute_elu(input, alpha=1.0, inplace=False):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(ivy.exp(input), 1),
    )
    ret = ivy.where(ivy.greater(input, 0), input, prod)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def _selu_with_inplace(input, inplace=False):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    prod = ivy.multiply(
        alpha,
        ivy.subtract(
            ivy.exp(input),
            1,
        ),
    )
    min_ = ivy.multiply(
        scale,
        ivy.minimum(0, prod),
    )
    max_ = ivy.multiply(
        scale,
        ivy.maximum(0, input),
    )
    ret = ivy.add(min_, max_)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def _rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    if training:
        # alpha = ivy.random_uniform(low=lower, high=upper)
        # ToDo implement alpha correctly after fixing ivy.random_uniform
        pass
    else:
        alpha = (lower + upper) / 2
    ret = ivy.subtract(
        ivy.relu(input), ivy.multiply(alpha, ivy.relu(ivy.negative(input)))
    )
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def kernels(ind, outd):
    def start_index(a, b, c):
        return math.floor((float(a) * float(c)) / b)

    def end_index(a, b, c):
        return math.ceil((float(a + 1) * float(c)) / b)

    results = []
    for ow in range(outd):
        start = start_index(ow, outd, ind)
        end = end_index(ow, outd, ind)
        sz = end - start
        results.append((start, sz))
    return results


def kernel_indexes(ind, out):
    startsLengths = kernels(ind, out)
    return [list(range(start, start + length)) for (start, length) in startsLengths]


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def sigmoid(input):
    return ivy.sigmoid(input)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def leaky_relu(input, negative_slope=0.01, inplace=False):
    ret = ivy.leaky_relu(input, alpha=negative_slope)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def gelu(
    input,
):  # , *, approximate="none"): ToDo: approximate is added in in PyTorch 1.12.1
    # if approximate == "none":
    # approximate = False
    # else:
    # approximate = True
    return ivy.gelu(input, approximate=False)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def tanh(input):
    return ivy.tanh(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def logsigmoid(input):
    return ivy.negative(ivy.softplus(ivy.negative(input)))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def softmin(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(-input, axis=dim)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def threshold(input, threshold, value, inplace=False):
    return _compute_threshold(input, threshold, value, inplace)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def threshold_(input, threshold, value):
    return _compute_threshold(input, threshold, value, inplace=True)


def relu6(input, inplace=False):
    ret = ivy.minimum(ivy.maximum(input, 0), 6)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def elu(input, alpha=1.0, inplace=False):
    return _compute_elu(input, alpha, inplace=inplace)


def elu_(input, alpha=1.0):
    return _compute_elu(input, alpha, inplace=True)


def celu(input, alpha=1.0, inplace=False):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(
            ivy.exp(ivy.divide(input, alpha)),
            1,
        ),
    )
    ret = ivy.add(
        ivy.maximum(0, input),
        ivy.minimum(0, prod),
    )
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def mish(input, inplace=False):
    ret = ivy.multiply(
        input,
        ivy.tanh(ivy.softplus(input)),
    )
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
def relu(input, inplace=False):
    ret = ivy.relu(input)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def relu_(input):
    ret = ivy.relu(input)
    ivy.inplace_update(input, ret)
    return input


def selu(input, inplace=False):
    return _selu_with_inplace(input, inplace=inplace)


@to_ivy_arrays_and_back
def prelu(input, weight):
    return ivy.add(ivy.maximum(0, input), ivy.multiply(weight, ivy.minimum(0, input)))


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    return _rrelu(input, lower, upper, training, inplace)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def rrelu_(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    return _rrelu(input, lower, upper, training, inplace=True)


@to_ivy_arrays_and_back
def hardshrink(input, lambd=0.5):
    mask = ivy.logical_or(ivy.greater(input, lambd), ivy.less(input, -lambd))
    return ivy.where(mask, input, 0.0)


@to_ivy_arrays_and_back
def softsign(input):
    return ivy.divide(input, ivy.add(1, ivy.abs(input)))


@to_ivy_arrays_and_back
def softshrink(input, lambd=0.5):
    low = ivy.where(ivy.less(input, -lambd), ivy.add(input, lambd), 0)
    up = ivy.where(ivy.greater(input, lambd), ivy.subtract(input, lambd), 0)
    return ivy.add(low, up)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def silu(input, inplace=False):
    ret = ivy.multiply(input, ivy.sigmoid(input))
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
def glu(input, dim=-1):
    a, b = ivy.split(input, num_or_size_splits=2, axis=dim)
    return ivy.multiply(a, ivy.sigmoid(b))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    if dim is None:
        dim = -1
    return ivy.log_softmax(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def tanhshrink(input):
    return ivy.subtract(input, ivy.tanh(input))


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def leaky_relu_(input, negative_slope=0.01):
    ret = ivy.leaky_relu(input, alpha=negative_slope)
    ivy.inplace_update(input, ret)
    return input


def hardswish(input, inplace=False):
    relu6_val = ivy.minimum(ivy.maximum(ivy.add(input, 3), 0), 6)
    ret = ivy.multiply(input, ivy.divide(relu6_val, 6))
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def hardsigmoid(input, inplace=False):
    ret = ivy.divide(ivy.minimum(ivy.maximum(ivy.add(input, 3), 0), 6), 6)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    less = ivy.where(ivy.less(input, min_val), min_val, input)
    ret = ivy.where(ivy.greater(input, max_val), max_val, less).astype(input.dtype)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def hardtanh_(input, min_val=-1.0, max_val=1.0):
    less = ivy.where(ivy.less(input, min_val), min_val, input)
    ret = ivy.where(ivy.greater(input, max_val), max_val, less).astype(input.dtype)
    ivy.inplace_update(input, ret)
    return input


@to_ivy_arrays_and_back
def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    abs_square = ivy.pow(ivy.abs(input), p)
    sum_ = ivy.sum(abs_square, axis=dim, keepdims=True)
    pnorm_res = ivy.pow(sum_, 1.0 / p)
    max_ = ivy.maximum(pnorm_res, eps)
    return ivy.divide(input, max_, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    shape = ivy.shape(input)
    if isinstance(normalized_shape, int) and normalized_shape == shape[-1]:
        axis = [-1]
    else:
        assert normalized_shape == shape[-len(normalized_shape) :]
        axis = list(range(len(shape) - len(normalized_shape), len(shape)))
    return ivy.layer_norm(input, axis, scale=weight, b=bias, epsilon=eps)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def softplus(input, beta=1, threshold=20):
    return ivy.softplus(input, beta=beta, threshold=threshold)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    shape = ivy.shape(input)
    assert shape[1] % num_groups == 0
    groups = shape[1] // num_groups
    num_dims = ivy.get_num_dims(input)
    expand_dims = (
        [0, *range(2, num_dims)] if weight is not None and num_dims > 2 else [0]
    )
    ret = ivy.concat(
        [
            ivy.layer_norm(
                input[:, i * groups : (i + 1) * groups, ...],
                list(range(1, num_dims)),
                scale=ivy.expand_dims(
                    weight[i * groups : (i + 1) * groups], axis=expand_dims
                )
                if weight is not None
                else None,
                b=ivy.expand_dims(bias[i * groups : (i + 1) * groups], axis=expand_dims)
                if bias is not None
                else None,
                epsilon=eps,
            )
            for i in range(num_groups)
        ],
        axis=1,
    )

    return ret


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
def batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    if training:
        dim = 0 if len(input.shape) == 2 else (0, 2, 3)
        current_mean = ivy.mean(input, axis=dim)
        current_var = ivy.var(input, axis=dim)
    else:
        current_mean = running_mean
        current_var = running_var

    input = ivy.swapaxes(input, 1, -1)
    input -= current_mean
    input /= ivy.sqrt(current_var + eps)
    if weight is not None:
        input *= weight
    if bias is not None:
        input += bias

    # updating running mean & var is useless in functional API?
    running_mean = (1.0 - momentum) * running_mean + momentum * current_mean
    running_var = (1.0 - momentum) * running_var + momentum * current_var

    return ivy.swapaxes(input, 1, -1)


# Reference: https://stackoverflow.com/a/63603993
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
def adaptive_avg_pool1d(input, output_size):
    squeeze = False
    if len(input.shape) == 2:
        input = ivy.expand_dims(input, axis=0)
        squeeze = True
    elif len(input.shape) != 3:
        raise ivy.exceptions.IvyException(
            f"Got {len(input.shape)}D input, but only 2D and 3D inputs are supported.",
        )
    input_size = input.shape[-1]
    if input_size % output_size == 0:
        stride = input_size // output_size
        kernel_size = input_size - (output_size - 1) * stride
        pooled_output = ivy.avg_pool1d(
            input, kernel_size, stride, "VALID", data_format="NCW"
        )
        if squeeze:
            return ivy.squeeze(pooled_output, axis=0)
        return pooled_output
    else:
        kernels = kernel_indexes(input_size, output_size)
        pooled_output = ivy.stack(
            [sum([input[:, :, x] for x in xs]) / len(xs) for xs in kernels], axis=-1
        )
        if squeeze:
            return ivy.squeeze(pooled_output, axis=0)
        return pooled_output


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_avg_pool2d(input, output_size):

    device = input.device
    shape = input.shape
    squeeze = False

    if len(input.shape) == 3:
        input = ivy.expand_dims(input, axis=0)
        squeeze = True
    elif len(input.shape) != 4:
        raise ivy.exceptions.IvyException(
            f"Got {len(shape)}D input, but only 3D and 4D inputs are supported.",
        )
    for d in input.shape[-2:]:
        if d == 0:
            raise ivy.exceptions.IvyException(
                "Expected input to have non-zero size for non-batch dimensions, but"
                f" input has shape {tuple(shape)}."
            )

    if all(i_s % o_s == 0 for i_s, o_s in zip(shape[-2:], output_size)):
        stride = tuple(i_s // o_s for i_s, o_s in zip(shape[-2:], output_size))
        kernel_size = tuple(
            i_s - (o_s - 1) * st
            for i_s, o_s, st in zip(shape[-2:], output_size, stride)
        )
        pooled_output = ivy.avg_pool2d(
            input, kernel_size, stride, "VALID", data_format="NCHW"
        )
        if squeeze:
            return ivy.squeeze(pooled_output, axis=0)
        return pooled_output

    def start_index(a, b, c):
        return ivy.trunc_divide(a * c, b).astype(ivy.int64)

    def end_index(a, b, c):
        return ivy.trunc_divide((a + 1) * c + b - 1, b).astype(ivy.int64)

    def compute_idx(in_size, out_size):
        orange = ivy.arange(out_size, device=device, dtype=ivy.int64)
        i0 = start_index(orange, out_size, in_size)
        maxlength = in_size // out_size + 1
        in_size_mod = in_size % out_size
        # adaptive = True iff there are kernels with different lengths
        adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
        if adaptive:
            maxlength += 1
        elif in_size_mod == 0:
            maxlength -= 1
        range_max = ivy.arange(maxlength, device=device, dtype=ivy.int64)
        idx = ivy.expand_dims(i0, axis=-1) + range_max
        if adaptive:
            maxval = ivy.full_like(idx, fill_value=in_size - 1)
            idx = ivy.minimum(idx, maxval)
            i1 = end_index(orange, out_size, in_size)
            length = i1 - i0
        else:
            length = maxlength
        return idx, length, range_max, adaptive

    def _expand_to_dim(x, dim):
        for _ in range(dim - len(x.shape)):
            x = ivy.expand_dims(x, axis=-1)
        return x

    idxh, length_h, range_max_h, adaptive_h = compute_idx(shape[-2], output_size[-2])
    idxw, length_w, range_max_w, adaptive_w = compute_idx(shape[-1], output_size[-1])

    # to numpy and back in order to bypass a slicing error in tensorflow
    vals = ivy.array(input.to_numpy()[..., _expand_to_dim(idxh, 4), idxw])

    if not adaptive_h and not adaptive_w:
        return ivy.mean(vals, axis=(-3, -1))

    def maybe_mask(vals, length, range_max, dim):
        if isinstance(length, int):
            return vals, length
        else:
            assert dim < 0
            mask = ivy.greater_equal(range_max, ivy.expand_dims(length, axis=-1))
            if dim == -2:
                mask = _expand_to_dim(mask, 4)
            vals = ivy.where(mask, 0.0, vals)
            length = _expand_to_dim(length, -dim)
            return vals, length

    vals, length_h = maybe_mask(vals, length_h, range_max_h, dim=-2)
    vals, length_w = maybe_mask(vals, length_w, range_max_w, dim=-1)

    ret = None
    for i, j in product(range(vals.shape[-3]), range(vals.shape[-1])):
        if ret is None:
            ret = vals[..., i, :, j]
        else:
            ret = ret + vals[..., i, :, j]
    pooled_output = ret / (length_h * length_w)

    if squeeze:
        return ivy.squeeze(pooled_output, axis=0)
    return pooled_output
