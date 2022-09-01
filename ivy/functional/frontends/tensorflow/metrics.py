import ivy

def sparse_categorical_accuracy(
        y_true,
        y_pred,
        sample_weight=None
):
    y_true = ivy.array(y_true)
    y_pred = ivy.array(y_pred)

    if sample_weight == None:
        sample_weight: ndarray = ivy.ones(y_true.shape)
    count = ivy.sum(sample_weight)
    total = ivy.dot(sample_weight, ivy.equal(y_true, ivy.argmax(y_pred, axis=1)))

    return ivy.divide(total, count)


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
