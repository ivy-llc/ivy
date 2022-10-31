import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def cosine_similarity(x1, x2, *, dim=1, eps=1e-08):
    x1, x2 = torch_frontend.promote_types_of_torch_inputs(x1, x2)

    if len(x1.shape) == len(x2.shape) and len(x2.shape) >= 2:
        numerator = ivy.sum(x1 * x2, axis=dim)
        x1_squared_norm = ivy.sum(ivy.square(x1), axis=dim)
        x2_squared_norm = ivy.sum(ivy.square(x2), axis=dim)
    else:
        numerator = ivy.sum(x1 * x2)
        x1_squared_norm = ivy.sum(ivy.square(x1))
        x2_squared_norm = ivy.sum(ivy.square(x2))

    x1_norm = ivy.sqrt(x1_squared_norm)
    x2_norm = ivy.sqrt(x2_squared_norm)
    norm_mm = x1_norm * x2_norm
    norm_mm, eps = torch_frontend.promote_types_of_torch_inputs(norm_mm, eps)
    denominator = ivy.maximum(norm_mm, eps)

    cosine = numerator / denominator
    return cosine
