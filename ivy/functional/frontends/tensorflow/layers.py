import ivy


def MAE(a, b):
    return ivy.mean(abs(a - b))