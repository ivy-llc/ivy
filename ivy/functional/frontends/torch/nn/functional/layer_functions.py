import ivy
from ivy.func_wrapper import with_supported_device_and_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


# --- Helpers --- #
# --------------- #


def _extract_states(states, batch_sizes):
    h = []
    for i in range(states.shape[1]):
        h.append(states[int(batch_sizes[i] - 1), i])
    h = ivy.expand_dims(ivy.stack(h, axis=0), axis=0)
    return h


def _lstm_full(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    ret = ivy.lstm(
        input,
        hx,
        params,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first=batch_first,
        has_ih_bias=has_biases,
        has_hh_bias=has_biases,
    )
    return ret[1], ret[2][0], ret[2][1]


def _lstm_packed(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    ret = ivy.lstm(
        data,
        hx,
        params,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_sizes=batch_sizes,
        has_ih_bias=has_biases,
        has_hh_bias=has_biases,
    )
    return ret[1], ret[2][0], ret[2][1]


# --- Main --- #
# ------------ #


@with_supported_device_and_dtypes(
    {"2.2 and below": {"cpu": ("float32", "float64")}},
    "torch",
)
@to_ivy_arrays_and_back
def lstm(*args, **kwargs):
    if "batch_sizes" in kwargs or (len(args) >= 4 and not isinstance(args[3], bool)):
        return _lstm_packed(*args, **kwargs)
    else:
        return _lstm_full(*args, **kwargs)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
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
