import ivy
import ivy.functional.frontends.paddle as paddle_frontend
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def pairwise_distance(x1, x2, *, p=2.0, eps=1e-06, keepdim=False):
    x1, x2 = paddle_frontend.promote_types_of_paddle_inputs(x1, x2)
    x1_dim = len(x1.shape)
    x2_dim = len(x2.shape)
    if x1_dim > x2_dim:
        output_dim = x1_dim
    else:
        output_dim = x2_dim

    return ivy.vector_norm(x1 - x2 + eps, ord=p, axis=output_dim - 1, keepdims=keepdim)
