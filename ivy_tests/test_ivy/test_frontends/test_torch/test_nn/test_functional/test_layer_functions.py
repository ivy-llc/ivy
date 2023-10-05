# global
from hypothesis import strategies as st

# local
import ivy
from ivy.functional.backends.torch.layers import _get_embed_dim
from ivy_tests.test_ivy import helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_nn.test_layers import _mha_helper


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
