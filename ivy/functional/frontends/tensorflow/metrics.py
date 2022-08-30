import ivy


def sparse_categorical_accuracy(y_true, y_pred):
    return ivy.cast(ivy.equal(ivy.max(y_true, axis=-1),
                              ivy.cast(ivy.argmax(y_pred, axis=-1), ivy.default_float_dtype(as_native=True))),
                    ivy.default_float_dtype(as_native=True))

def sparse_categorical_crossentropy(
    y_true,
    y_pred,
    from_logits=False,
    axis=-1
):
    if from_logits:
        y_pred = ivy.softmax(y_pred)
    return ivy.sparse_cross_entropy(y_true, y_pred, axis=axis)


sparse_categorical_crossentropy.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
