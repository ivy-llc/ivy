import ivy


def binary_accuracy(y_true, y_pred, threshold=0.5):
    ivy.equal(y_true, y_pred > threshold)
    return ivy.mean(ivy.equal(y_true, y_pred > threshold))
