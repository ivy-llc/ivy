import ivy
import ivy.functional.frontends.torch as torch_frontend


def cosine_similarity(x1, x2, dim=1, eps=1e-08):
    eps_2 = eps * eps
    x1_squared_norm = ivy.pow(x1, 2).sum(axis=dim, keepdims=True)
    x2_squared_norm = ivy.pow(x2, 2).sum(axis=dim, keepdims=True)
    x1_squared_norm, x2_squared_norm = torch_frontend.promote_types_of_torch_inputs(
        x1_squared_norm, x2_squared_norm
    )
    x1_x2_squared_norm = x1_squared_norm * x2_squared_norm
    eps_2, x1_x2_squared_norm = torch_frontend.promote_types_of_torch_inputs(
        eps_2, x1_x2_squared_norm
    )
    x1_x2_squared_norm = ivy.maximum(x1_x2_squared_norm, eps_2)
    x1_x2_norm = ivy.squeeze(ivy.sqrt(x1_x2_squared_norm), axis=dim)
    x1, x2 = torch_frontend.promote_types_of_torch_inputs(x1, x2)
    x1_x2 = ivy.sum(x1 * x2, axis=dim)
    x1_x2, x1_x2_norm = torch_frontend.promote_types_of_torch_inputs(x1_x2, x1_x2_norm)
    return ivy.divide(x1_x2, x1_x2_norm)
