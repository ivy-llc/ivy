import ivy


def pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False):
    return ivy.vector_norm(x1 - x2 + eps, axis=-1, ord=p, keepdims=keepdim)
