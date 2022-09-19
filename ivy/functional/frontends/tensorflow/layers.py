import ivy


def MAE(y_pred, y_true):
    return ivy.mean(abs(y_pred - y_true))
    