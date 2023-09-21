import torch
import xformers.ops as xops
from ivy.func_wrapper import to_native_arrays_and_back


@to_native_arrays_and_back
def scaled_dot_product_attention(
    q,
    k,
    v,
    scale: float,
    /,
    *,
    mask=None,
    out=None,
):
    if isinstance(mask, torch.Tensor):
        mask = torch.where(mask == 0, -torch.inf, 0)
    return xops.memory_efficient_attention(q, k, v, scale=scale, attn_bias=mask)
