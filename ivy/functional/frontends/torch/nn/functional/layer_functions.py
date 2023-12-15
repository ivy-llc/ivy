import ivy
from ivy.func_wrapper import with_supported_device_and_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.ivy.experimental.manipulation import _slice_along_axis
from ivy.utils.exceptions import IvyNotImplementedException


# --- Helpers --- #
# --------------- #


def _extract_states(states, batch_sizes):
    h = []
    for i in range(states.shape[1]):
        h.append(states[int(batch_sizes[i] - 1), i])
    h = ivy.expand_dims(ivy.stack(h, axis=0), axis=0)
    return h


def _generic_lstm(
    input,
    initial_states,
    all_weights,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first=False,
    batch_sizes=None,
):
    weights_per_layer = 4 if has_biases else 2

    assert len(all_weights) == num_layers * weights_per_layer * (1 + bidirectional)
    layer_weights = [
        all_weights[i : i + weights_per_layer]
        for i in range(0, len(all_weights), weights_per_layer)
    ]

    if batch_sizes is not None:
        input, batch_sizes = _pad_packed_sequence(input, batch_sizes)

    if batch_first:
        input = ivy.swapaxes(input, 0, 1)

    if dropout and train:
        raise IvyNotImplementedException()

    unidirectional = not bidirectional

    h0, c0 = initial_states
    h_outs, c_outs = [], []

    output = input
    for i in range(num_layers):
        if unidirectional:
            if weights_per_layer == 4:
                weight_ih, weight_hh, (bias_i, bias_h) = _transform_weights(
                    layer_weights, i
                )
            else:
                weight_ih, weight_hh = _transform_weights_no_bias(layer_weights, i)
                bias_i = bias_h = None

            state_indices = i, i + 1
        else:
            if weights_per_layer == 4:
                weight_ih_f, weight_hh_f, (bias_i_f, bias_h_f) = _transform_weights(
                    layer_weights, 2 * i
                )
                weight_ih_b, weight_hh_b, (bias_i_b, bias_h_b) = _transform_weights(
                    layer_weights, 2 * i + 1
                )
            else:
                weight_ih_f, weight_hh_f = _transform_weights_no_bias(
                    layer_weights, 2 * i
                )
                weight_ih_b, weight_hh_b = _transform_weights_no_bias(
                    layer_weights, 2 * i + 1
                )
                bias_i_f = bias_h_f = bias_i_b = bias_h_b = None

            weight_ih = weight_ih_f, weight_ih_b
            weight_hh = weight_hh_f, weight_hh_b
            bias_i = bias_i_f, bias_i_b
            bias_h = bias_h_f, bias_h_b

            state_indices = 2 * i, 2 * i + 2

        output, (h_out, c_out) = _lstm_layer(
            output,
            (
                _retrieve_state(h0, *state_indices, num_layers),
                _retrieve_state(c0, *state_indices, num_layers),
            ),
            (weight_ih, weight_hh),
            (bias_i, bias_h),
            bidirectional,
            batch_first=batch_first,
            batch_sizes=batch_sizes,
        )
        h_outs.append(h_out)
        c_outs.append(c_out)

    if batch_first:
        output = ivy.swapaxes(output, 0, 1)

    h_outs = h_out if num_layers == 1 else ivy.concat(h_outs, axis=0)
    c_outs = c_out if num_layers == 1 else ivy.concat(c_outs, axis=0)

    if batch_sizes is not None:
        output = _pack_padded_sequence(output, batch_sizes)[0]

    return output, h_outs, c_outs


def _lstm_cell(
    x,
    init_h,
    init_c,
    kernel,
    recurrent_kernel,
    bias,
    recurrent_bias,
    batch_first,
    batch_sizes=None,
):
    init_h = ivy.squeeze(init_h, axis=0)
    init_c = ivy.squeeze(init_c, axis=0)
    return ivy.lstm_update(
        x,
        init_h,
        init_c,
        kernel,
        recurrent_kernel,
        bias=bias,
        recurrent_bias=recurrent_bias,
        time_major=not batch_first,
    )


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
    return _generic_lstm(
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first=batch_first,
    )


def _lstm_layer(
    x, hidden, weights, biases, bidirectional, batch_first, batch_sizes=None
):
    if not bidirectional:
        result, (h, c) = _lstm_cell(
            x,
            *hidden,
            *weights,
            *biases,
            batch_first=batch_first,
            batch_sizes=batch_sizes,
        )
    else:
        result_fw, (h_fw, c_fw) = _lstm_cell(
            x,
            hidden[0][:1],
            hidden[1][:1],
            weights[0][0],
            weights[1][0],
            biases[0][0],
            biases[1][0],
            batch_first=batch_first,
            batch_sizes=batch_sizes,
        )
        x_reversed = ivy.flip(x, axis=0)
        result_bw, (h_bw, c_bw) = _lstm_cell(
            x_reversed,
            hidden[0][1:],
            hidden[1][1:],
            weights[0][1],
            weights[1][1],
            biases[0][1],
            biases[1][1],
            batch_first=batch_first,
            batch_sizes=batch_sizes,
        )
        result_bw = ivy.flip(result_bw, axis=0)
        result = ivy.concat([result_fw, result_bw], axis=len(result_fw.shape) - 1)
        c = ivy.concat([c_fw, c_bw], axis=0)
        h = ivy.concat([h_fw, h_bw], axis=0)
    return result, (h, c)


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
    return _generic_lstm(
        data,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_sizes=batch_sizes,
    )


def _pack_padded_sequence(input, lengths):
    input = ivy.swapaxes(input, 0, 1)
    data = []
    batch_sizes = []
    for i in range(int(max(lengths))):
        valid_data_mask = ivy.array(lengths) > i
        data.append(input[valid_data_mask, i])
        batch_sizes.append(int(sum(valid_data_mask)))
    data = ivy.concat(data)
    batch_sizes = ivy.array(batch_sizes, dtype=ivy.int64)
    return data, batch_sizes


def _pad_packed_sequence(data, batch_sizes):
    padded_data = ivy.full(
        (len(batch_sizes), int(max(batch_sizes)), *data.shape[1:]),
        0,
        dtype=data.dtype,
        device=data.device,
    )
    data_offset = 0
    for i, batch_size in enumerate(batch_sizes):
        batch_size = int(batch_size)
        padded_data[i, :batch_size] = data[data_offset : data_offset + batch_size]
        data_offset += batch_size
    lengths = ivy.sum(
        ivy.arange(1, int(max(batch_sizes)) + 1)[:, ivy.newaxis] <= batch_sizes,
        axis=1,
        dtype=ivy.int64,
    )
    return padded_data, lengths


def _retrieve_state(x, start, end, num_layers):
    return x if num_layers == 1 else _slice_along_axis(x, start=start, stop=end, axis=0)


def _transform_weights(layer_weights, layer_index):
    weights = layer_weights[layer_index]
    weight_ih, weight_hh, bias_ih, bias_hh = weights
    return (
        ivy.swapaxes(weight_ih, 0, 1),
        ivy.swapaxes(weight_hh, 0, 1),
        (bias_ih, bias_hh),
    )


def _transform_weights_no_bias(layer_weights, layer_index):
    weights = layer_weights[layer_index]
    weight_ih, weight_hh = weights
    return ivy.swapaxes(weight_ih, 0, 1), ivy.swapaxes(weight_hh, 0, 1)


# --- Main --- #
# ------------ #


@with_supported_device_and_dtypes(
    {"2.1.1 and below": {"cpu": ("float32", "float64")}},
    "torch",
)
@to_ivy_arrays_and_back
def lstm(*args, **kwargs):
    if "batch_sizes" in kwargs or (len(args) >= 4 and not isinstance(args[3], bool)):
        return _lstm_packed(*args, **kwargs)
    else:
        return _lstm_full(*args, **kwargs)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.1.1 and below": ("float32", "float64")}, "torch")
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
