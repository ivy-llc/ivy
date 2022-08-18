import ivy


def sparse_categorical_crossentropy(
    y_true,
    y_pred,
    from_logits=False,
    axis=-1
):
    if from_logits:
        y_pred = ivy.softmax(y_pred)
    return ivy.sparse_cross_entropy(y_true, y_pred, axis=axis, epsilon=1e-7)


sparse_categorical_crossentropy.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
