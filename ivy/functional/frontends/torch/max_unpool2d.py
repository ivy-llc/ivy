# -*- coding: utf-8 -*-


import torch

def max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    """
    Unpools the input tensor using the provided indices from the corresponding max_pool2d operation.

    Args:
        input (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        indices (torch.Tensor): Indices tensor obtained from the corresponding max_pool2d operation.
            It should have the same shape as the input tensor.
        kernel_size (int or tuple): Size of the max pooling window.
        stride (int or tuple): Stride of the max pooling window. Default: kernel_size.
        padding (int or tuple): Zero-padding added to both sides of the input. Default: 0.
        output_size (tuple): The desired output size of the unpooling operation.
            If specified, it overrides the output size calculation based on the other parameters. Default: None.

    Returns:
        torch.Tensor: Unpooled tensor of shape (batch_size, channels, output_height, output_width).
    """
    if stride is None:
        stride = kernel_size

    if output_size is None:
        output_size = (
            (input.size(2) - 1) * stride + kernel_size - 2 * padding,
            (input.size(3) - 1) * stride + kernel_size - 2 * padding
        )

    batch_size, channels, height, width = input.size()
    output_height, output_width = output_size

    # Create an empty tensor with the desired output size
    output = torch.zeros(batch_size, channels, output_height, output_width, device=input.device)

    # Iterate over each element in the input tensor
    for b in range(batch_size):
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    # Get the index for the max value in the pooling window
                    index = indices[b, c, h, w]

                    # Compute the unpooled indices based on the index and stride
                    unpool_h = (h * stride[0]) + (index // output_width)
                    unpool_w = (w * stride[1]) + (index % output_width)

                    # Set the value in the output tensor at the unpooled indices
                    output[b, c, unpool_h, unpool_w] = input[b, c, h, w]

    return output