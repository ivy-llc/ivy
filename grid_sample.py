def grid_sample(input, grid, mode='bilinear', padding_mode='zeros'):
    if mode == 'nearest':
        # Round the grid values to get the closest integer indices
        x_rounded = ivy.round(grid[..., 0])
        y_rounded = ivy.round(grid[..., 1])

        if padding_mode == 'zeros':
            # Create masks for out-of-bound x and y positions
            mask_x = ivy.logical_or(x_rounded < 0, x_rounded >= input.shape[-1])
            mask_y = ivy.logical_or(y_rounded < 0, y_rounded >= input.shape[-2])

            # Combine the masks
            mask = ivy.logical_or(mask_x, mask_y)

            # Using the indices, gather the values from the input tensor
            sampled_output = ivy.where(mask, ivy.zeros_like(input), input[..., y_rounded, x_rounded])

        elif padding_mode == 'border':
            # Clamp the indices to lie within the borders
            x_clamped = ivy.clip(x_rounded, 0, input.shape[-1] - 1)
            y_clamped = ivy.clip(y_rounded, 0, input.shape[-2] - 1)

            # Using the clamped indices, gather the values from the input tensor
            sampled_output = input[..., y_clamped, x_clamped]

        else:
            raise ValueError("Unsupported padding_mode. Expected 'zeros' or 'border'.")

    elif mode == 'bilinear':
        # Bilinear interpolation
        raise NotImplementedError("Bilinear interpolation has not been implemented yet.")

    else:
        raise ValueError("Unsupported mode. Expected 'bilinear' or 'nearest'.")

    return sampled_output
