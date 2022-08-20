import ivy


def binary_accuracy(y_true, y_pred, threshold=0.5):
    return ivy.mean(
        ivy.equal(y_true, ivy.astype(y_pred > threshold, y_pred.dtype)).astype(
            y_pred.dtype
        ),
        axis=-1,
    )
