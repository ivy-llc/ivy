# global
from hypothesis import assume, strategies as st
import numpy as np

# local
import ivy
from ivy.functional.ivy.layers import _get_embed_dim, _pack_padded_sequence
from ivy_tests.test_ivy import helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_nn.test_layers import _mha_helper


# --- Helpers --- #
# --------------- #


@st.composite
def _lstm_helper(draw):
    dtype = draw(helpers.get_dtypes("valid", full=False))

    has_biases = draw(st.booleans())
    bidirectional = draw(st.booleans())
    dropout = draw(st.floats(min_value=0, max_value=0.99))
    train = (
        draw(st.booleans()) and not dropout
    )  # not yet supported by original function
    packed = draw(st.booleans())

    batch_first = draw(st.booleans()) and not packed
    num_batches = draw(st.integers(min_value=1, max_value=5))
    num_layers = draw(st.integers(min_value=1, max_value=3))
    num_directions = 2 if bidirectional else 1
    seq_size = draw(st.integers(min_value=1, max_value=5))
    in_size = draw(st.integers(min_value=1, max_value=3))
    hidden_size = draw(st.integers(min_value=1, max_value=3))

    input = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(
                (num_batches, seq_size, in_size)
                if batch_first
                else (seq_size, num_batches, in_size)
            ),
            min_value=0,
            max_value=1,
        )
    )

    init_h = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(num_directions * num_layers, num_batches, hidden_size),
            min_value=0,
            max_value=1,
        )
    )
    init_c = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(num_directions * num_layers, num_batches, hidden_size),
            min_value=0,
            max_value=1,
        )
    )

    all_weights = []
    for k in range(num_layers):
        for _ in range(num_directions):
            weight_ih = draw(
                helpers.array_values(
                    dtype=dtype[0],
                    shape=(
                        (4 * hidden_size, in_size)
                        if k == 0
                        else (4 * hidden_size, num_directions * hidden_size)
                    ),
                    min_value=0,
                    max_value=1,
                )
            )
            weight_hh = draw(
                helpers.array_values(
                    dtype=dtype[0],
                    shape=(4 * hidden_size, hidden_size),
                    min_value=0,
                    max_value=1,
                )
            )
            all_weights += [weight_ih, weight_hh]
            if has_biases:
                bias_ih = draw(
                    helpers.array_values(
                        dtype=dtype[0],
                        shape=(4 * hidden_size,),
                        min_value=0,
                        max_value=1,
                    )
                )
                bias_hh = draw(
                    helpers.array_values(
                        dtype=dtype[0],
                        shape=(4 * hidden_size,),
                        min_value=0,
                        max_value=1,
                    )
                )
                all_weights += [bias_ih, bias_hh]

    if packed:
        batch_sizes = [seq_size]
        batch_sizes += draw(
            st.lists(
                st.integers(min_value=1, max_value=seq_size),
                min_size=num_batches - 1,
                max_size=num_batches - 1,
            )
        )
        batch_sizes = np.array(draw(st.permutations(batch_sizes)))
        input, batch_sizes = (
            ivy.to_numpy(p) for p in _pack_padded_sequence(input, batch_sizes)
        )
    else:
        batch_sizes = None

    initial_states = init_h, init_c
    all_weights = tuple(all_weights)
    if batch_sizes is not None:
        dtypes = dtype + ["int64"]
        kwargs = {
            "data": input,
            "batch_sizes": batch_sizes,
            "hx": initial_states,
            "params": all_weights,
            "has_biases": has_biases,
            "num_layers": num_layers,
            "dropout": dropout,
            "train": train,
            "bidirectional": bidirectional,
        }
    else:
        dtypes = dtype
        kwargs = {
            "input": input,
            "hx": initial_states,
            "params": all_weights,
            "has_biases": has_biases,
            "num_layers": num_layers,
            "dropout": dropout,
            "train": train,
            "bidirectional": bidirectional,
            "batch_first": batch_first,
        }
    return dtypes, kwargs


# --- Main --- #
# ------------ #


# lstm
@handle_frontend_test(
    fn_tree="torch.lstm",
    dtypes_kwargs=_lstm_helper(),
    test_with_out=st.just(False),
)
def test_torch_lstm(
    *,
    dtypes_kwargs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtypes, kwargs = dtypes_kwargs
    # Todo: Debug the function to have this case passing as well
    assume("batch_sizes" not in kwargs)
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **kwargs,
    )


# multi_head_attention_forward
@handle_frontend_test(
    fn_tree="torch.nn.functional.multi_head_attention_forward",
    dtype_mha_args=_mha_helper(same_pre_embed_dim=True, batch_second=True).filter(
        lambda args: args[10] is not None
        and (not args[22] or args[5] is not None)
        and len(set(_get_embed_dim(*args[6:10], args[1]))) == 1
    ),
    test_with_out=st.just(False),
)
def test_torch_multi_head_attention_forward(
    *,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    dtype_mha_args,
    backend_fw,
):
    (
        dtype,
        q,
        k,
        v,
        heads,
        attn_mask,
        in_proj_weight,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        out_proj_weight,
        in_proj_bias,
        out_proj_bias,
        key_padding_mask,
        bias_k,
        bias_v,
        static_k,
        static_v,
        _,
        add_zero_attn,
        dropout_p,
        training,
        is_causal,
        need_weights,
        average_attn_weights,
        batch_first,
    ) = dtype_mha_args
    if k is None and v is None:
        k = v = q
    # re-order the dtypes to match the order of the frontend arguments, not the order
    # of ivy.multi_head_attention's arguments given by _mha_helper
    kwargs = {
        "query": q,
        "key": k,
        "value": v,
        "embed_dim_to_check": q.shape[-1],
        "num_heads": heads,
        "in_proj_weight": in_proj_weight,
        "in_proj_bias": in_proj_bias,
        "bias_k": bias_k,
        "bias_v": bias_v,
        "add_zero_attn": add_zero_attn,
        "dropout_p": dropout_p,
        "out_proj_weight": out_proj_weight,
        "out_proj_bias": out_proj_bias,
        "training": training,
        "key_padding_mask": key_padding_mask,
        "need_weights": need_weights,
        "attn_mask": attn_mask,
        "use_separate_proj_weight": in_proj_weight is None,
        "q_proj_weight": q_proj_weight,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "static_k": static_k,
        "static_v": static_v,
        "average_attn_weights": average_attn_weights,
        "is_causal": is_causal,
    }
    helpers.test_frontend_function(
        input_dtypes=[str(r.dtype) for r in kwargs.values() if ivy.is_array(r)],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        atol=1e-03,
        on_device=on_device,
        test_values=not training or dropout_p == 0.0,
        **kwargs,
    )
