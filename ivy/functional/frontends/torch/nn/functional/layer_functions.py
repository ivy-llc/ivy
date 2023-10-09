import ivy
from ivy import with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.ivy.experimental.manipulation import _slice_along_axis
from ivy.utils.exceptions import IvyNotImplementedException


# --- Helpers --- #
# --------------- #


def _generic_lstm(
    input,
    initial_states,
    all_weights,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first=None,
    batch_sizes=None,
):
    weights_per_layer = 4 if has_biases else 2

    assert len(all_weights) == num_layers * weights_per_layer * (1 + bidirectional)
    layer_weights = [
        all_weights[i : i + weights_per_layer]
        for i in range(0, len(all_weights), weights_per_layer)
    ]

    if batch_sizes is not None:
        input = _pad_packed_sequence(input, batch_sizes, batch_first=batch_first)

    if batch_first:
        input = ivy.permute_dims(input, axes=(1, 0, 2))

    if dropout and train:
        raise IvyNotImplementedException()

    w_hh = all_weights[1]
    hidden_size = w_hh.shape[1]

    unidirectional = not bidirectional

    h_outs = []

    h0, c0 = initial_states
    c_outs = []

    # pytorch is input, forget, cell, output.
    # onnx is    input, output, forget, cell.
    reform_permutation = [(0, 1), (3, 4), (1, 3)]

    def reform_weights(w, n, intervals):
        slices = [
            _slice_along_axis(w, start=x * n, stop=y * n, axis=0) for x, y in intervals
        ]
        return ivy.concat(slices, axis=0)

    def transform_weights_no_bias(layer_index):
        weights = layer_weights[layer_index]
        weight_ih, weight_hh = (
            reform_weights(w, hidden_size, reform_permutation) for w in weights
        )
        return tuple(ivy.expand_dims(x, axis=0) for x in (weight_ih, weight_hh))

    def transform_weights(layer_index):
        weights = layer_weights[layer_index]
        weight_ih, weight_hh, bias_ih, bias_hh = (
            reform_weights(w, hidden_size, reform_permutation) for w in weights
        )
        bias_concat = ivy.concat([bias_ih, bias_hh], axis=0)
        return tuple(
            ivy.expand_dims(x, axis=0) for x in (weight_ih, weight_hh, bias_concat)
        )

    def retrieve_state(x, start, end):
        return (
            x
            if num_layers == 1
            else _slice_along_axis(x, start=start, stop=end, axis=0)
        )

    def lstm_single_layer(x, hidden, weights, bias):
        if batch_first:
            x = ivy.swapaxes(x, 0, 1)

        hx_fw, cx_fw = hidden
        if bidirectional:
            if hx_fw is None:
                hx_bw = None
            else:
                hx_bw = hx_fw[1]
                hx_fw = hx_fw[0]
            if cx_fw is None:
                cx_bw = None
            else:
                cx_bw = cx_fw[1]
                cx_fw = cx_fw[0]
            hidden_bw = hx_bw, cx_bw
        hidden_fw = hx_fw, cx_fw
        result_fw, hidden_fw = ivy.lstm_update(x, *hidden_fw, *weights, bias=bias)

        if bidirectional:
            x_reversed = ivy.flip(x, axis=0)
            result_bw, hidden_bw = ivy.lstm_update(
                x_reversed, *hidden_bw, *weights, bias=bias
            )
            result_bw = ivy.flip(result_bw, axis=0)

            result = ivy.concat([result_fw, result_bw], axis=len(result_fw.shape) - 1)
            if hidden_fw is None and hidden_bw is None:
                h = None
                c = None
            elif hidden_fw is None:
                h, c = hidden_bw
            elif hidden_bw is None:
                h, c = hidden_fw
            else:
                h = ivy.stack([hidden_fw[0], hidden_bw[0]], axis=0)
                c = ivy.stack([hidden_fw[1], hidden_bw[1]], axis=0)
        else:
            result = result_fw
            h, c = hidden_fw

        if batch_first:
            result = ivy.swapaxes(result, 0, 1)

        return result, (h, c)

    for i in range(num_layers):
        if unidirectional:
            if weights_per_layer == 4:
                weight_ih, weight_hh, bias_concat = transform_weights(i)
            else:
                weight_ih, weight_hh = transform_weights_no_bias(i)
                bias_concat = None

            state_indices = i, i + 1
        else:
            if weights_per_layer == 4:
                weight_ih_f, weight_hh_f, bias_f = transform_weights(2 * i)
                weight_ih_b, weight_hh_b, bias_b = transform_weights(2 * i + 1)
                bias_concat = ivy.concat([bias_f, bias_b], axis=0)
            else:
                weight_ih_f, weight_hh_f = transform_weights_no_bias(2 * i)
                weight_ih_b, weight_hh_b = transform_weights_no_bias(2 * i + 1)
                bias_concat = None

            weight_ih = ivy.concat([weight_ih_f, weight_ih_b], axis=0)
            weight_hh = ivy.concat([weight_hh_f, weight_hh_b], axis=0)

            state_indices = 2 * i, 2 * i + 2

        output, (h_out, c_out) = lstm_single_layer(
            input,
            (retrieve_state(h0, *state_indices), retrieve_state(c0, *state_indices)),
            (weight_ih, weight_hh),
            bias_concat,
        )

        if bidirectional:
            # The ONNX RNN/GRU/LSTM produce an output of dimensions
            #   seq_len, num_directions, batch, hidden_size
            # We have to convert to match pytorch's expected
            #   seq_len, batch, num_directions * hidden_size
            # by first moving num_directions before hidden_size with
            # Transpose, and then combining it with hidden_size
            # with Reshape.
            output = ivy.permute_dims(output, axes=(0, 2, 1, 3))
            output = ivy.reshape(output, (*output.shape[:2], -1))
        else:
            output = ivy.squeeze(output, axis=1)

        h_outs.append(h_out)

    if batch_first:
        output = ivy.permute_dims(output, axes=(1, 0, 2))
    h_outs = h_out if num_layers == 1 else ivy.concat(h_outs, axis=0)
    c_outs = c_out if num_layers == 1 else ivy.concat(c_outs, axis=0)
    return output, h_outs, c_outs


def _lstm_full(
    input,
    hidden_v,
    weight_v,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    return _generic_lstm(
        input,
        hidden_v,
        weight_v,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first=batch_first,
    )


def _lstm_packed(
    input,
    batch_sizes,
    hidden_v,
    weight_v,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    return _generic_lstm(
        input,
        hidden_v,
        weight_v,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_sizes=batch_sizes,
    )


# used in testing
def _pack_padded_sequence(padded_sequence, lengths, batch_first=False):
    if not batch_first:
        padded_sequence = ivy.swapaxes(padded_sequence, 0, 1)
    data = []
    for i, length in enumerate(lengths):
        data += [padded_sequence[i, :length]]
    data = ivy.concat(data)
    return data


def _pad_packed_sequence(data, batch_sizes, batch_first=False, padding_value=0):
    padded_sequence = ivy.full(
        (len(batch_sizes), max(batch_sizes), data.shape[-1]),
        padding_value,
        dtype=data.dtype,
        device=data.device,
    )
    data_pointer = 0
    for i, batch_size in enumerate(batch_sizes):
        padded_sequence[i, :batch_size] = data[data_pointer : data_pointer + batch_size]
        data_pointer += batch_size
    if not batch_first:
        padded_sequence = ivy.swapaxes(padded_sequence, 0, 1)
    return padded_sequence


# --- Main --- #
# ------------ #


@to_ivy_arrays_and_back
def lstm(*args, **kwargs):
    if "batch_sizes" in kwargs or (len(args) >= 4 and not isinstance(args[3], bool)):
        return _lstm_packed(*args, **kwargs)
    else:
        return _lstm_full(*args, **kwargs)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.1.0 and below": ("float32", "float64")}, "torch")
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
