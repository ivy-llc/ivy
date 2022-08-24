import ivy


def binary_matches(y_true, y_pred, threshold=0.5):
    y_pred = ivy.array(y_pred)
    threshold = ivy.astype(ivy.array(threshold), y_pred.dtype)
    y_pred = ivy.astype(ivy.greater(y_pred, threshold), y_pred.dtype)
    return ivy.astype(
        ivy.equal(y_true, y_pred), ivy.default_float_dtype(input=y_pred, as_native=True)
    )


def binary_accuracy(y_true, y_pred, threshold=0.5):
    return ivy.mean(binary_matches(y_true, y_pred, threshold), axis=-1)
