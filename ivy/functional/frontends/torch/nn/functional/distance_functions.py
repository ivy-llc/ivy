import ivy


def cosine_similarity(x1, x2, dim=1, eps=1e-08):
    x1 = ivy.flatten(x1)
    x2 = ivy.flatten(x2)
    return ivy.vecdot(x1, x2, axis=dim) / ivy.maximum(
        ivy.vector_norm(x1) * ivy.vector_norm(x2), eps
    )
