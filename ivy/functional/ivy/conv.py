import ivy
import ivy.numpy as np  # Import the NumPy backend for Ivy


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
# Define the modified conv3d_transpose function
def conv3d_transpose(
    x: ivy.Array,
    filters: ivy.Array,
    strides: int,
    padding: str,
    /,
    *,
    output_shape: ivy.Shape = None,
    data_format: str = "NDHWC",
    dilations: int = 1,
    bias: ivy.Array = None,
    out: ivy.Array = None,
) -> ivy.Array:
    """
    Compute a 3-D transpose convolution.

    Computes a 3-D transpose convolution operation on the input `x` using the provided `filters`.

    Parameters
    ----------
    x : ivy.Array
        Input volume of shape `[batch_size, d, h, w, d_in]` or `[batch_size, d_in, d, h, w]`.
    filters : ivy.Array
        Convolution filters of shape `[fd, fh, fw, d_in, d_out]`.
    strides : int
        The stride of the sliding window for each dimension of the input.
    padding : str
        Either 'SAME' (padding so that the output's shape is the same as the input's) or
        'VALID' (padding so that the output's shape is specified by `output_shape`).
    output_shape : ivy.Shape, optional
        Shape of the output (Default value = None).
    data_format : str, optional
        The ordering of the dimensions in the input, one of "NDHWC" or "NCDHW". "NDHWC"
        corresponds to inputs with shape (batch_size, depth, height, width, channels),
        while "NCDHW" corresponds to input with shape (batch_size, channels, depth,
        height, width) (Default value = "NDHWC").
    dilations : int, optional
        The dilation factor for each dimension of input (Default value = 1).
    bias : ivy.Array, optional
        Bias array of shape `[d_out]`.
    out : ivy.Array, optional
        Optional output array for writing the result to. It must have a shape that
        matches the input's shape.

    Returns
    -------
    ivy.Array
        The result of the transpose convolution operation.

    Examples
    --------
    # Sample input data
    x = ivy.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3])
    filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 6])

    # Call the modified conv3d_transpose function
    result = conv3d_transpose(x, filters, 2, 'SAME')

    # Print the shape of the result
    print(result.shape)
    """

    return ivy.current_backend(x).conv3d_transpose(
        x,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        bias=bias,
        out=out,
    )


# Sample input data
x = ivy.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3])
filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 6])

# Call the modified conv3d_transpose function
result = conv3d_transpose(x, filters, 2, "SAME")

# Print the shape of the result
print(result.shape)
