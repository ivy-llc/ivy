import ivy
import paddle
from ivy.func_wrapper import ivy # Import the mean_paddle function from your code
import numpy as np
def mean_paddle(x):
    """
    Calculate the mean of a PaddlePaddle tensor.

    Parameters:
    x (paddle.Tensor): The input tensor.

    Returns:
    paddle.Tensor: The mean of the input tensor.
    """
    return paddle.mean(x)

@ivy.mean
def mean_paddle(x):
    return ivy.mean(x)


 