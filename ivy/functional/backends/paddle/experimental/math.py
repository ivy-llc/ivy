# ivy/paddle/tensor/math.py

import paddle

def ceil_(x):
    """
    Compute the ceiling value of the input tensor element-wise.

    Args:
        x (ivy.ndarray): Input tensor.

    Returns:
        ivy.ndarray: Tensor with the ceiling value of each element of x.

    Example:
        >>> import ivy
        >>> x = ivy.array([1.4, 2.7, 3.2])
        >>> ivy.ceil_(x)
        array([2., 3., 4.], dtype=float32)

    """
    # Implement the ceil_ function using the Paddle backend.
    # This function should call the corresponding Paddle function to compute the ceiling value.
    return paddle.ceil(x)
