# global
# from typing import List, Optional
import ivy
# import torch


def sigmoid(input, out=None):
    return ivy.sigmoid(input, out=out)


sigmoid.unsupported_dtypes = ("float16",)


def softmax(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


softmax.unsupported_dtypes = ("float16",)


# def layer_norm(
#     input: torch.Tensor,
#     normalized_shape: List[int],
#     weight: Optional[torch.Tensor] = None,
#     bias: Optional[torch.Tensor] = None,
#     eps: float = ivy._MIN_BASE,
#     new_std: Optional[float] = 1.0,
#     out: Optional[torch.Tensor] = None
# ) -> torch.Tensor:
#     return ivy.layer_norm(
#         x=input,
#         normalized_idxs=normalized_shape,
#         epsilon=eps,
#         scale=weight,
#         offset=bias,
#         new_std=new_std,
#         out=out
#     )


# layer_norm.unsupported_dtypes = ("float16",)
