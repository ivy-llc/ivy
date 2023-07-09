import paddle

def create_affine_grid(theta, size):
    # theta: Affine transformation parameters (batch_size, 2, 3)
    # size: Size of the output grid (batch_size, channels, height, width)

    batch_size = size[0]
    grid = paddle.nn.functional.affine_grid(theta, size)

    return grid

# Example usage
theta = paddle.to_tensor([[[1, 0, 0.2], [0, 1, 0.3]]])  # Affine transformation matrix
size = (1, 1, 64, 64)  # Output grid size

grid = create_affine_grid(theta, size)
print(grid.shape)  # Output: [1, 64, 64, 2]

