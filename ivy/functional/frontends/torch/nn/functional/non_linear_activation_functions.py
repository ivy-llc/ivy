# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


# --- Helpers --- #
# --------------- #


def _compute_elu(input, alpha=1.0):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(ivy.exp(input), 1),
    )
    return ivy.where(ivy.greater(input, 0), input, prod)


def _compute_threshold(input, threshold, value):
    return ivy.where(ivy.greater(input, threshold), input, value)


def _rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    if training:
        # alpha = ivy.random_uniform(low=lower, high=upper)
        # ToDo implement alpha correctly after fixing ivy.random_uniform
        pass
    else:
        alpha = (lower + upper) / 2
    return ivy.subtract(
        ivy.relu(input), ivy.multiply(alpha, ivy.relu(ivy.negative(input)))
    )


def _selu_with_inplace(input):
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
    return ivy.add(min_, max_)


# --- Main --- #
# ------------ #


@to_ivy_arrays_and_back
def celu(input, alpha=1.0, inplace=False):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(
            ivy.exp(ivy.divide(input, alpha)),
            1,
        ),
    )
    return ivy.add(
        ivy.maximum(0, input),
        ivy.minimum(0, prod),
    )


@to_ivy_arrays_and_back
def elu(input, alpha=1.0, inplace=False):
    return _compute_elu(input, alpha)


def elu_(input, alpha=1.0):
    return elu(input, alpha=alpha, inplace=True)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
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
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    gumbels = -ivy.empty_like(logits).exponential().log()
    gumbels = (logits + gumbels) / tau
    y_soft = ivy.softmax(x=gumbels, axis=dim)

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
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    less = ivy.where(ivy.less(input, min_val), min_val, input)
    return ivy.where(ivy.greater(input, max_val), max_val, less).astype(input.dtype)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def hardtanh_(input, min_val=-1.0, max_val=1.0):
    return hardtanh(input, min_val=min_val, max_val=max_val, inplace=True)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def leaky_relu(input, negative_slope=0.01, inplace=False):
    return ivy.leaky_relu(input, alpha=negative_slope)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def leaky_relu_(input, negative_slope=0.01):
    ret = ivy.leaky_relu(input, alpha=negative_slope)
    ivy.inplace_update(input, ret)
    return input


@to_ivy_arrays_and_back
def local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0):
    dim = len(ivy.shape(input))
    if dim < 3:
        raise ValueError(
            "Expected 3D or higher dimensionality input (got {} dimensions)".format(dim)
        )
    if input.size == 0:
        return input
    div = ivy.multiply(input, input)

    if dim == 3:
        div = ivy.expand_dims(div, axis=1)
        div = ivy.zero_pad(div, ((0, 0), (0, 0), (size // 2, (size - 1) // 2), (0, 0)))
        div = ivy.avg_pool2d(
            div, (size, 1), 1, "VALID", count_include_pad=True, data_format="NCHW"
        )
        div = ivy.squeeze(div, axis=1)
    else:
        sizes = ivy.shape(input)
        div = ivy.reshape(div, (sizes[0], 1, sizes[1], sizes[2], -1))
        div = ivy.zero_pad(
            div, ((0, 0), (0, 0), (size // 2, (size - 1) // 2), (0, 0), (0, 0))
        )
        div = ivy.avg_pool3d(
            div, (size, 1, 1), 1, "VALID", count_include_pad=True, data_format="NCDHW"
        )
        div = ivy.squeeze(div, axis=1)
        div = ivy.reshape(div, sizes)

    div = ivy.pow(ivy.add(ivy.multiply(div, alpha), k), beta)
    return ivy.divide(input, div)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    if dim is None:
        dim = -1
    return ivy.log_softmax(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
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
@with_supported_dtypes({"2.0.1 and below": ("float32", "float64")}, "torch")
def multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    average_attn_weights=True,
    is_causal=False,
):
    embed_dim = query.shape[-1]
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    return ivy.multi_head_attention(
        query,
        key=key,
        value=value,
        batch_first=False,
        num_heads=num_heads,
        attention_mask=attn_mask,
        in_proj_weights=in_proj_weight if not use_separate_proj_weight else None,
        q_proj_weights=q_proj_weight,
        k_proj_weights=k_proj_weight,
        v_proj_weights=v_proj_weight,
        out_proj_weights=out_proj_weight,
        in_proj_bias=in_proj_bias,
        out_proj_bias=out_proj_bias,
        is_causal=is_causal and not (need_weights or key_padding_mask is not None),
        key_padding_mask=key_padding_mask,
        bias_k=bias_k,
        bias_v=bias_v,
        static_k=static_k,
        static_v=static_v,
        add_zero_attn=add_zero_attn,
        return_attention_weights=need_weights,
        average_attention_weights=average_attn_weights,
        dropout=dropout_p,
        training=training,
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
@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, "torch")
def relu6(input, inplace=False):
    return ivy.relu6(input)


def relu_(input):
    return relu(input, inplace=True)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    return _rrelu(input, lower, upper, training)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def rrelu_(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    return rrelu(input, lower=lower, upper=upper, training=training, inplace=True)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.0.1 and below": ("float32", "float64")}, "torch")
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
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def sigmoid(input):
    return ivy.sigmoid(input)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def silu(input, inplace=False):
    return ivy.multiply(input, ivy.sigmoid(input))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def softmin(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(-input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
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
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def tanh(input):
    return ivy.tanh(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def tanhshrink(input):
    return ivy.subtract(input, ivy.tanh(input))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def threshold(input, threshold, value, inplace=False):
    return _compute_threshold(input, threshold, value)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def threshold_(input, threshold, value):
    return threshold(input, threshold, value, inplace=True)
