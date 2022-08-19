import ivy


def sparse_categorical_crossentropy(
    y_true,
    y_pred,
    from_logits=False,
    axis=-1
):
    if from_logits:
        ivy.softmax(y_pred, out=y_pred)
    return ivy.sparse_cross_entropy(y_true, y_pred, axis=axis)


sparse_categorical_crossentropy.unsupported_dtypes = {"torch": ("float16", "bfloat16")}

