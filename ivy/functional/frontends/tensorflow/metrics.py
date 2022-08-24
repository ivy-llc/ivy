import ivy


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


def mean_absolute_error(y_true, y_pred):
    return ivy.mean(ivy.abs(y_true-y_pred))

mean_absolute_error.unsupported_dtypes = {"torch": ("float16", "bfloat16"),
                                          "numpy": ("float16","bfloat16", "float32", "float64"),
                                          }

