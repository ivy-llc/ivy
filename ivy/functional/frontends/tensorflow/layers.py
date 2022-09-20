import ivy
from metrics import mean_absolute_error


def MAE(y_pred, y_true):
    return mean_absolute_error(y_true - y_pred)
