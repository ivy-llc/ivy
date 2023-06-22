def corrcoef(x, y):
    """
    Args:
        x (paddle.Tensor): The first variable.
        y (paddle.Tensor): The second variable.
    """
    correlation_matrix = paddle.fluid.layers.correlation(x, y)
    correlation_coefficient = correlation_matrix.numpy()[0, 1]
    return correlation_coefficient
