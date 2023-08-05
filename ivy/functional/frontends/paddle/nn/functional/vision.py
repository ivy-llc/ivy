# local

import ivy,math
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
def affine_grid(theta, out_shape, align_corners=True):
    if len(out_shape) == 4:
        N, C, H, W = out_shape
        base_grid = ivy.empty((N, H, W, 3))
        if align_corners:
            base_grid[:, :, :, 0] = ivy.linspace(-1, 1, W)
            base_grid[:, :, :, 1] = ivy.expand_dims(ivy.linspace(-1, 1, H), axis=-1)
            height_values = ivy.expand_dims(ivy.linspace(-1, 1, H), axis=-1)
            base_grid[:, :, :, 1] = ivy.array(
                [[[height_values[i]] * W for i in range(H)]]
            )[:, :, :, 0]
            base_grid[:, :, :, 2] = ivy.full((H, W), 1)
            grid = ivy.matmul(base_grid.view((N, H * W, 3)), theta.swapaxes(1, 2))
            return grid.view((N, H, W, 2))
        else:
            base_grid[:, :, :, 0] = ivy.linspace(-1, 1, W) * (W - 1) / W
            base_grid[:, :, :, 1] = ivy.expand_dims(
                ivy.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            height_values = ivy.expand_dims(
                ivy.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            base_grid[:, :, :, 1] = ivy.array(
                [[[height_values[i]] * W for i in range(H)]]
            )[:, :, :, 0]
            base_grid[:, :, :, 2] = ivy.full((H, W), 1)
        grid = ivy.matmul(base_grid.view((N, H * W, 3)), ivy.swapaxes(theta, 1, 2))
        return grid.view((N, H, W, 2))
    else:
        N, C, D, H, W = out_shape
        base_grid = ivy.empty((N, D, H, W, 4))
        if align_corners:
            base_grid[:, :, :, :, 0] = ivy.linspace(-1, 1, W)
            base_grid[:, :, :, :, 1] = ivy.expand_dims(ivy.linspace(-1, 1, H), axis=-1)
            height_values = ivy.linspace(-1, 1, H)
            base_grid[:, :, :, :, 1] = ivy.array(
                [[[[height_values[i]] * W for i in range(H)]] * D]
            )
            base_grid[:, :, :, :, 2] = ivy.expand_dims(
                ivy.expand_dims(ivy.linspace(-1, 1, D), axis=-1), axis=-1
            )
            width_values = ivy.linspace(-1, 1, D)
            base_grid[:, :, :, :, 2] = ivy.array(
                [[ivy.array([[width_values[i]] * W] * H) for i in range(D)]]
            )
            base_grid[:, :, :, :, 3] = ivy.full((D, H, W), 1)
            grid = ivy.matmul(base_grid.view((N, D * H * W, 4)), theta.swapaxes(1, 2))
            return grid.view((N, D, H, W, 3))
        else:
            base_grid[:, :, :, :, 0] = ivy.linspace(-1, 1, W) * (W - 1) / W
            base_grid[:, :, :, :, 1] = ivy.expand_dims(
                ivy.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            height_values = ivy.linspace(-1, 1, H) * (H - 1) / H
            base_grid[:, :, :, :, 1] = ivy.array(
                [[[[height_values[i]] * W for i in range(H)]] * D]
            )
            base_grid[:, :, :, :, 2] = ivy.expand_dims(
                ivy.expand_dims(ivy.linspace(-1, 1, D) * (D - 1) / D, axis=-1), axis=-1
            )
            width_values = ivy.linspace(-1, 1, D) * (D - 1) / D
            base_grid[:, :, :, :, 2] = ivy.array(
                [[ivy.array([[width_values[i]] * W] * H) for i in range(D)]]
            )
            base_grid[:, :, :, :, 3] = ivy.full((D, H, W), 1)
            grid = ivy.matmul(base_grid.view((N, D * H * W, 4)), theta.swapaxes(1, 2))
            return grid.view((N, D, H, W, 3))
#grid_sample 
def grid_sample(input, grid, mode, padding_mode):

    '''grid_sample is used to perfome various spatial transformations.
    Once you have the grid of sampling points, you use the grid_sample function to resample the pixel values from the original image to the new positions defined by the grid.
    
    In simple terms, the function is used for transforming images or data from one coordinate system to another. 
    
    Parameters:
    - input: input tensor data/image of shape(batch_size, channels, input_height, input_width) that needs to be transformed. 

    - grid: typically created using affine_grid, it defines the transformation to be applied to the input tensor. 
            It is essentially a set of coordinates that dictate how the output tensor will be computed from the input tensor. 

    - mode: determines the interpolation method used to estimate pixel values at non-integer coordinates when performing the sampling operation.
            In simpler terms, the mode arg. is like choosing a method to guess the colour of new pixels when we resize or transform an image.

    - padding_mode: optional parameter that controls how to handle out-of-bound values during the interpolation process.
                    It specifies the padding strategy to be used when sampling points outside the input image boundaries. 
   '''

    if mode not in ['nearest','bilinear']:
        raise ValueError("Invalid mode. Supported modes are 'nearest' and 'bilinear'. ")

    if padding_mode not in ['zeros', 'border', 'reflection']:
        raise ValueError("Invalid padding mode. Supported modes are 'zeros', 'border' and 'reflection'. ")
    
    if len(grid.shape) != 4 or grid.shape[3] != 2:
        raise ValueError("The grid should be a 4D tensor with the last dim having size 2.") # size 2 representing (x,y) coordinates. 
    
    if mode == 'nearest':
        return grid_sample_nearest(input, grid, padding_mode)
    elif mode == 'bilinear':
        return grid_sample_bilinear(input, grid, padding_mode)

def grid_sample_nearest(input, grid, padding_mode):

    '''Assigns the pixel value at non-integer grid coordinates to be the value of the nearest pixel of the input data.
    The transformation process involves normalising the grid & rounding them to the nearest pixel value.   
    
    parameters: 
    - input: the input data/image that needs to be transformed. 
    - grid: the transformation grid, defines the transformation to be applied to the input tensor. 
    - padding_mode: controls how to handle out-of-bound values during the interpolation process. 
    
    Returns: 
    - gathered tensor: the transformed values for each point in the output grid. These transformed values represent the result of applying the spatial transformation. '''

    # get the spatial dims of the input & grid
    input_shape = ivy.shape(input)
    grid_shape = ivy.shape(grid)
    batch, channels, in_height, in_width = input_shape
    out_height, out_width = grid_shape[1], grid_shape[2]

    # normalise grid coordinates to the range [-1, 1]
    normalised_grid = 2.0 * grid / ivy.array([in_width -1, in_height - 1])
    normalised_grid = normalised_grid - ivy.array([1.0, 1.0])
    
    # map the normalised grid coordinates to the output tensor using round operation
    if padding_mode == 'zeros':
        grid_x = ivy.clip(math.round(normalised_grid[:,:,:,0]), -1.0, 1.0)
        grid_y = ivy.clip(math.round(normalised_grid[:,:,:,1]), -1.0, 1.0)
    else:
        grid_x = ivy.clip(math.round(normalised_grid[:,:,:,0]), 0.0, 1.0)
        grid_y = ivy.clip(math.round(normalised_grid[:,:,:,0]), 0.0, 1.0)
    
    # convert the grid coordinaates to indices
    x_indices = ((grid_x + 1.0) * 0.5 * (in_width - 1)).astype('int32')
    y_indices = ((grid_y + 1.0) * 0.5 * (in_height -1)).astype('int32')
    # Gather values from input using indices
    gathered = input[:, :, y_indices, x_indices]

    return gathered # the shape of the gathered tensor is (batch, channels, out_height, out_width)

def grid_sample_bilinear(input, grid, padding_mode):
    '''
    The bilinear interpolation calculates the pixel value at a non-integer grid coordinate as a weighted average of the pixel values of the four nearest neighbouring pixels.
    The interpolation is performed based on the relative distances of the grid coordinates to these neighbouring pixels. Pixels closer to the grid coordinate have higher weights, while pixels farther away have lower weights. 
    
    Parameters: 
    - input: the image/data that needs to be transformed. 
    - grid: the transformation grid, defines the transformation to be applied to the input tensor. 
    - padding_mode: optional paramter that controls how to handle out-of-bounds values during the interpolation process. 
    
    Returns: 
    The transformed values representing the result of applying the spatial transformation. 
     '''
    # Get the spatial dimensions of the input and grid
    input_shape = ivy.shape(input)
    grid_shape = ivy.shape(grid)
    batch_size, channels, in_height, in_width = input_shape
    out_height, out_width = grid_shape[1], grid_shape[2]

    # normalise grid cordinates to the range [-1, 1]
    normalised_grid = 2.0 * grid / ivy.array([in_width - 1, in_height - 1])
    normalised_grid = normalised_grid - ivy.array([1.0, 1.0])

    # map the normalised grid coordinates to the input tensor
    if padding_mode == 'zeros':
        grid_x = ivy.clip((normalised_grid[:, :, :, 0] + 1.0) * 0.5 * (in_width - 1), 0.0, in_width - 1)
        grid_y = ivy.clip((normalised_grid[:, :, :, 1] + 1.0) * 0.5 * (in_height - 1), 0.0, in_height - 1)
    else:
        grid_x = ivy.clip((normalised_grid[:, :, :, 0] + 1.0) * 0.5 * in_width, 0.0, in_width - 1)
        grid_y = ivy.clip((normalised_grid[:, :, :, 1] + 1.0) * 0.5 * in_height, 0.0, in_height - 1)

    # Calculate the four corner points of the grid cells
    x0 = grid_x.floor().astype('int32')
    y0 = grid_y.floor().astype('int32')

    # Ensure that the grid_x and grid_y indices do not exceed input tensor dimensions
    x0 = ivy.clip(x0, 0, in_width - 1)
    y0 = ivy.clip(y0, 0, in_height - 1)
    x1 = ivy.clip(x0 + 1, 0, in_width - 1)
    y1 = ivy.clip(y0 + 1, 0, in_height - 1)
    
    # Calculate the relative distance of the grid coordinates from the corner points
    wx = grid_x - x0
    wy = grid_y - y0
    
    # Gather pixel values of the 4 corner points for each point in the output grid
    i00 = input[:, :, y0, x0] # top-left corner 
    i10 = input[:, :, y0, x1] # top-right 
    i01 = input[:, :, y1, x0] # bottom-left
    i11 = input[:, :, y1, x1] # bottomright
    
    #  bilinear interpolation
    interpolated = i00 * (1 - wx) * (1 - wy) + i10 * wx * (1 - wy) + i01 * (1 - wx) * wy + i11 * wx * wy
    
    return interpolated # the output shape of the gathered tensor is batch, n_channels, out_height, out_width 


import numpy as np
import matplotlib.pyplot as plt

def test_grid_sample_bilinear():
    # Create an example image (4x4) with some arbitrary pixel values
    input_image = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]])

    # Expand dimensions to create a batch and channel dimension
    input_image_batch = input_image[np.newaxis, np.newaxis, :, :]  

    # Create a grid of sampling points (output grid) with coordinates for transformation
    grid = np.array([[[[-1.0, -1.0], [1.0, 1.0]],
                      [[-0.5, -0.5], [0.5, 0.5]],
                      [[0.0, 0.0], [0.0, 0.0]],
                      [[0.5, 0.5], [-0.5, -0.5]]]])

    # Apply grid_sample_bilinear to transform the image
    transformed_image = grid_sample_bilinear(input_image_batch, grid, padding_mode='zeros')

    # Extract the transformed image from the result (remove batch and channel dimensions)
    transformed_image = transformed_image[0, 0, :, :]
    print(transformed_image.shape)

    # Display the original and transformed images side by side
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image.ravel(), cmap='gray')
    plt.title("Transformed Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Test the function
test_grid_sample_bilinear()



