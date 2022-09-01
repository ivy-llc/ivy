import ivy


def binary_matches(y_true, y_pred, threshold=0.5):
    y_pred = ivy.array(y_pred)
    threshold = ivy.astype(ivy.array(threshold), y_pred.dtype)
    y_pred = ivy.astype(ivy.greater(y_pred, threshold), y_pred.dtype)
    return ivy.astype(
        ivy.equal(y_true, y_pred), ivy.default_float_dtype(as_native=True)
    )


def binary_accuracy(y_true, y_pred, threshold=0.5):
    return ivy.mean(binary_matches(y_true, y_pred, threshold), axis=-1)


def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    if from_logits:
        y_pred = ivy.softmax(y_pred)
    return ivy.sparse_cross_entropy(y_true, y_pred, axis=axis)


sparse_categorical_crossentropy.unsupported_dtypes = {
    "numpy": ("float16", "bfloat16", "float32", "float64"),
}


def mean_absolute_error(y_true, y_pred):
    return ivy.mean(ivy.abs(y_true - y_pred))


mean_absolute_error.unsupported_dtypes = {
    "numpy": ("int8", "float64"),
    "torch": ("int8", "float64"),
}


def binary_crossentropy(
    y_true, y_pred, from_logits: bool = False, label_smoothing: float = 0.0
):
    if from_logits:
        y_pred = ivy.softmax(y_pred)
    return ivy.mean(ivy.binary_cross_entropy(y_true, y_pred, label_smoothing))

def mean_squared_logarithmic_error(y_true, y_pred):
    y_pred = ivy.asarray(y_pred)
    y_true = ivy.astype(y_true, y_pred.dtype)
    first_log = ivy.log(ivy.maximum(y_pred, 1e-7) + 1.0)
    second_log = ivy.log(ivy.maximum(y_true, 1e-7) + 1.0)
    return ivy.mean(ivy.square(ivy.subtract(first_log, second_log)), axis=-1)