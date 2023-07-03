# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


def _compute_threshold(input, threshold, value):
    return ivy.where(ivy.greater(input, threshold), input, value)


def _compute_elu(input, alpha=1.0):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(ivy.exp(input), 1),
    )
    return ivy.where(ivy.greater(input, 0), input, prod)


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


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def sigmoid(input):
    return ivy.sigmoid(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def leaky_relu(input, negative_slope=0.01, inplace=False):
    return ivy.leaky_relu(input, alpha=negative_slope)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


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
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def tanh(input):
    return ivy.tanh(input)


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
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def softmin(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(-input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def threshold(input, threshold, value, inplace=False):
    return _compute_threshold(input, threshold, value)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def threshold_(input, threshold, value):
    return threshold(input, threshold, value, inplace=True)


@to_ivy_arrays_and_back
def relu6(input, inplace=False):
    return ivy.relu6(input)


@to_ivy_arrays_and_back
def elu(input, alpha=1.0, inplace=False):
    return _compute_elu(input, alpha)


def elu_(input, alpha=1.0):
    return elu(input, alpha=alpha, inplace=True)


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
def mish(input, inplace=False):
    return ivy.multiply(
        input,
        ivy.tanh(ivy.softplus(input)),
    )


@to_ivy_arrays_and_back
def relu(input, inplace=False):
    return ivy.relu(input)


def relu_(input):
    return relu(input, inplace=True)


@to_ivy_arrays_and_back
def selu(input, inplace=False):
    return ivy.selu(input)


@to_ivy_arrays_and_back
def prelu(input, weight):
    return ivy.add(ivy.maximum(0, input), ivy.multiply(weight, ivy.minimum(0, input)))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    return _rrelu(input, lower, upper, training)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def rrelu_(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    return rrelu(input, lower=lower, upper=upper, training=training, inplace=True)


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


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def silu(input, inplace=False):
    return ivy.multiply(input, ivy.sigmoid(input))


@to_ivy_arrays_and_back
def glu(input, dim=-1):
    a, b = ivy.split(input, num_or_size_splits=2, axis=dim)
    return ivy.multiply(a, ivy.sigmoid(b))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    if dim is None:
        dim = -1
    return ivy.log_softmax(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def tanhshrink(input):
    return ivy.subtract(input, ivy.tanh(input))


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def leaky_relu_(input, negative_slope=0.01):
    ret = ivy.leaky_relu(input, alpha=negative_slope)
    ivy.inplace_update(input, ret)
    return input


@to_ivy_arrays_and_back
def hardswish(input, inplace=False):
    relu6_val = ivy.relu6(ivy.add(input, 3))
    return ivy.multiply(input, ivy.divide(relu6_val, 6))


@to_ivy_arrays_and_back
def hardsigmoid(input, inplace=False):
    return ivy.divide(ivy.minimum(ivy.maximum(ivy.add(input, 3), 0), 6), 6)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    less = ivy.where(ivy.less(input, min_val), min_val, input)
    return ivy.where(ivy.greater(input, max_val), max_val, less).astype(input.dtype)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def hardtanh_(input, min_val=-1.0, max_val=1.0):
    return hardtanh(input, min_val=min_val, max_val=max_val, inplace=True)


@to_ivy_arrays_and_back
def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    abs_square = ivy.pow(ivy.abs(input), p)
    sum_ = ivy.sum(abs_square, axis=dim, keepdims=True)
    pnorm_res = ivy.pow(sum_, 1.0 / p)
    max_ = ivy.maximum(pnorm_res, eps)
    return ivy.divide(input, max_, out=out)


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
    # q/k/v shape: (seq_len, batch_size, embed_dim)
    seq_len, batch_size, embed_dim = query.shape
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    assert key.shape == value.shape

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim needs to be divisible by heads"
    scale = ivy.sqrt(head_dim)

    if use_separate_proj_weight:
        assert key.shape[:2] == value.shape[:2], (
            f"key's sequence and batch dims {key.shape[:2]} do not match value's"
            f" {value.shape[:2]}"
        )
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    if is_causal and key_padding_mask is None and not need_weights:
        mask = ivy.tril(ivy.ones((seq_len, seq_len), dtype=query.dtype), k=0)
        attn_mask = ivy.zeros((seq_len, seq_len), dtype=query.dtype)
        attn_mask = ivy.where(mask == 0.0, float("-inf"), 0)

    if in_proj_bias is None:
        q_bias, k_bias, v_bias = None, None, None
    else:
        q_bias, k_bias, v_bias = ivy.split(in_proj_bias, num_or_size_splits=3)

    if not use_separate_proj_weight:
        q_proj_weight, k_proj_weight, v_proj_weight = ivy.split(
            in_proj_weight, num_or_size_splits=3
        )

    q = ivy.linear(query, q_proj_weight, bias=q_bias)
    k = ivy.linear(key, k_proj_weight, bias=k_bias)
    v = ivy.linear(value, v_proj_weight, bias=v_bias)

    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = ivy.concat([k, ivy.tile(bias_k, (1, batch_size, 1))])
        v = ivy.concat([v, ivy.tile(bias_v, (1, batch_size, 1))])
        if attn_mask is not None:
            attn_mask = ivy.concat(
                [attn_mask, ivy.zeros((attn_mask.shape[0], 1), dtype=attn_mask.dtype)],
                axis=1,
            )
        if key_padding_mask is not None:
            key_padding_mask = ivy.concat(
                [
                    key_padding_mask,
                    ivy.zeros(
                        (key_padding_mask.shape[0], 1), dtype=key_padding_mask.dtype
                    ).bool(),
                ],
                axis=1,
            )

    q = ivy.swapaxes(q.reshape((q.shape[0], batch_size * num_heads, head_dim)), 0, 1)

    if static_k is None:
        k = ivy.swapaxes(
            k.reshape((k.shape[0], batch_size * num_heads, head_dim)), 0, 1
        )
    else:
        assert static_k.shape[0] == batch_size * num_heads, (
            f"expecting static_k.shape[0] of {batch_size * num_heads}, but got"
            f" {static_k.shape[0]}"
        )
        assert (
            static_k.shape[2] == head_dim
        ), f"expecting static_k.shape[2] of {head_dim}, but got {static_k.shape[2]}"
        k = static_k

    if static_v is None:
        v = ivy.swapaxes(
            v.reshape((v.shape[0], batch_size * num_heads, head_dim)), 0, 1
        )
    else:
        assert static_v.shape[0] == batch_size * num_heads, (
            f"expecting static_v.shape[0] of {batch_size * num_heads}, but got"
            f" {static_v.shape[0]}"
        )
        assert (
            static_v.shape[2] == head_dim
        ), f"expecting static_v.shape[2] of {head_dim}, but got {static_v.shape[2]}"
        v = static_v

    # TODO add_zero_attn doesn't work for all cases
    # fix this and add test cases (by changing to add_zero_attn=st.booleans())
    if add_zero_attn:
        zero_attn_shape = (batch_size * num_heads, 1, head_dim)
        k = ivy.concat([k, ivy.zeros(zero_attn_shape, dtype=k.dtype)], axis=1)
        v = ivy.concat([v, ivy.zeros(zero_attn_shape, dtype=v.dtype)], axis=1)
        if attn_mask is not None:
            attn_mask = ivy.pad(attn_mask, [(0, 0), (0, 1)])
        if key_padding_mask is not None:
            key_padding_mask = ivy.pad(key_padding_mask, [(0, 0), (0, 1)])

    src_len = k.shape[1]
    attn_weights = ivy.matmul(q, ivy.swapaxes(k, 1, 2))
    assert list(attn_weights.shape) == [batch_size * num_heads, seq_len, src_len]

    attn_weights = attn_weights / scale

    if attn_mask is not None:
        attn_mask = ivy.expand_dims(attn_mask, axis=0)
        attn_weights += attn_mask

    if key_padding_mask is not None:
        key_padding_mask = ivy.expand_dims(
            ivy.expand_dims(key_padding_mask, axis=1), axis=2
        )
        attn_weights = attn_weights.reshape((batch_size, num_heads, seq_len, src_len))
        attn_weights = ivy.where(key_padding_mask < 0.0, float("-inf"), attn_weights)
        attn_weights = attn_weights.reshape((batch_size * num_heads, seq_len, src_len))

    attn_weights = ivy.softmax(attn_weights, axis=-1)
    attn_weights = ivy.dropout(attn_weights, dropout_p, training=training)

    attn_output = ivy.matmul(attn_weights, v)
    assert list(attn_output.shape) == [batch_size * num_heads, seq_len, head_dim]
    attn_output = ivy.swapaxes(attn_output, 0, 1).reshape(
        (seq_len, batch_size, embed_dim)
    )
    attn_output = ivy.linear(attn_output, out_proj_weight, bias=out_proj_bias)

    if need_weights:
        attn_weights = attn_weights.reshape((batch_size, num_heads, seq_len, src_len))
        if average_attn_weights:
            attn_weights = ivy.sum(attn_weights, axis=1) / num_heads
        return (attn_output, attn_weights)
    else:
        return (attn_output,)
