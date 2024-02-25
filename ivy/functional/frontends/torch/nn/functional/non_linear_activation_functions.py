# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "complex",
            "float16",
        )
    },
    "torch",
)
def celu(input, alpha=1.0, inplace=False):
    return ivy.celu(input, alpha=alpha)


def celu_(input, alpha=1.0):
    return celu(input, alpha=alpha, inplace=True)


@to_ivy_arrays_and_back
def elu(input, alpha=1.0, inplace=False):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(ivy.exp(input), 1),
    )
    return ivy.where(ivy.greater(input, 0), input, prod)


def elu_(input, alpha=1.0):
    return elu(input, alpha=alpha, inplace=True)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def gelu(input, *, approximate="none"):
    if approximate == "none":
        return ivy.gelu(input, approximate=False)
    elif approximate == "tanh":
        return ivy.gelu(input, approximate=True)
    else:
        raise ivy.utils.exceptions.IvyException(
            "`approximate` argument must be either 'none' or 'tanh'."
        )


@to_ivy_arrays_and_back
def glu(input, dim=-1):
    a, b = ivy.split(input, num_or_size_splits=2, axis=dim)
    return ivy.multiply(a, ivy.sigmoid(b))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    gumbels = -ivy.empty_like(logits).exponential().log()
    gumbels = (logits + gumbels) / tau
    y_soft = ivy.softmax(gumbels, axis=dim)

    if hard:
        indices = y_soft.max(axis=dim, keepdims=True)[1]
        y_hard = ivy.zeros_like(logits)
        updates = ivy.ones_like(indices)
        y_hard = ivy.scatter_nd(indices, updates, reduction="replace", out=y_hard)

        ret = y_hard - y_soft.stop_gradient(preserve_type=True) + y_soft
    else:
        ret = y_soft

    return ret


@to_ivy_arrays_and_back
def hardshrink(input, lambd=0.5):
    mask = ivy.logical_or(ivy.greater(input, lambd), ivy.less(input, -lambd))
    return ivy.where(mask, input, 0.0)


@to_ivy_arrays_and_back
def hardsigmoid(input, inplace=False):
    return ivy.divide(ivy.minimum(ivy.maximum(ivy.add(input, 3), 0), 6), 6)


@to_ivy_arrays_and_back
def hardswish(input, inplace=False):
    relu6_val = ivy.relu6(ivy.add(input, 3))
    return ivy.multiply(input, ivy.divide(relu6_val, 6))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    less = ivy.where(ivy.less(input, min_val), min_val, input)
    return ivy.where(ivy.greater(input, max_val), max_val, less).astype(input.dtype)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def hardtanh_(input, min_val=-1.0, max_val=1.0):
    return hardtanh(input, min_val=min_val, max_val=max_val, inplace=True)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def leaky_relu(input, negative_slope=0.01, inplace=False):
    return ivy.leaky_relu(input, alpha=negative_slope)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def leaky_relu_(input, negative_slope=0.01):
    return leaky_relu(input, negative_slope=negative_slope, inplace=True)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.2 and below": ("float",)}, "torch")
def local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0):
    non_batched = input.ndim == 3
    if non_batched:
        input = ivy.expand_dims(input, axis=2)
    ret = ivy.local_response_norm(
        input, size, bias=k, alpha=alpha, beta=beta, average=True, data_format="NCHW"
    )
    if non_batched:
        ret = ivy.squeeze(ret, axis=2)
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    if dim is None:
        dim = -1
    return ivy.log_softmax(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def logsigmoid(input):
    return ivy.logsigmoid(input)


@to_ivy_arrays_and_back
def mish(input, inplace=False):
    return ivy.multiply(
        input,
        ivy.tanh(ivy.softplus(input)),
    )


@to_ivy_arrays_and_back
def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    abs_square = ivy.pow(ivy.abs(input), p)
    sum_ = ivy.sum(abs_square, axis=dim, keepdims=True)
    pnorm_res = ivy.pow(sum_, 1.0 / p)
    max_ = ivy.maximum(pnorm_res, eps)
    return ivy.divide(input, max_, out=out)


@to_ivy_arrays_and_back
def prelu(input, weight):
    return ivy.add(ivy.maximum(0, input), ivy.multiply(weight, ivy.minimum(0, input)))


@to_ivy_arrays_and_back
def relu(input, inplace=False):
    return ivy.relu(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("complex",)}, "torch")
def relu6(input, inplace=False):
    return ivy.relu6(input)


@to_ivy_arrays_and_back
def relu_(input):
    return relu(input, inplace=True)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    if training:
        # alpha = ivy.random_uniform(low=lower, high=upper)
        # ToDo implement alpha correctly after fixing ivy.random_uniform
        pass
    else:
        alpha = (lower + upper) / 2
    return ivy.subtract(
        ivy.relu(input), ivy.multiply(alpha, ivy.relu(ivy.negative(input)))
    )


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def rrelu_(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    return rrelu(input, lower=lower, upper=upper, training=training, inplace=True)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    return ivy.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
        mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


@to_ivy_arrays_and_back
def selu(input, inplace=False):
    return ivy.selu(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def sigmoid(input):
    return ivy.sigmoid(input)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def silu(input, inplace=False):
    return ivy.multiply(input, ivy.sigmoid(input))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def softmin(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(-input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def softplus(input, beta=1, threshold=20):
    return ivy.softplus(input, beta=beta, threshold=threshold)


@to_ivy_arrays_and_back
def softshrink(input, lambd=0.5):
    low = ivy.where(ivy.less(input, -lambd), ivy.add(input, lambd), 0)
    up = ivy.where(ivy.greater(input, lambd), ivy.subtract(input, lambd), 0)
    return ivy.add(low, up)


@to_ivy_arrays_and_back
def softsign(input):
    return ivy.divide(input, ivy.add(1, ivy.abs(input)))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def tanh(input):
    return ivy.tanh(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def tanhshrink(input):
    return ivy.subtract(input, ivy.tanh(input))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def threshold(input, threshold, value, inplace=False):
    return ivy.where(ivy.greater(input, threshold), input, value)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
def threshold_(input, threshold, value):
    return threshold(input, threshold, value, inplace=True)
