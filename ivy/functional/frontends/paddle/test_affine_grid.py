import paddle
import numpy as np

def test_create_affine_grid():
    # Define test inputs
    theta = paddle.to_tensor([[[1, 0, 0.2], [0, 1, 0.3]]])
    size = (1, 1, 64, 64)

    # Generate expected output
    expected_shape = [1, 64, 64, 2]
    expected_grid = np.zeros(expected_shape)
    for i in range(expected_shape[1]):
        for j in range(expected_shape[2]):
            expected_grid[0, i, j, 0] = j + 0.2  # x-coordinate
            expected_grid[0, i, j, 1] = i + 0.3  # y-coordinate

    # Call the function
    grid = create_affine_grid(theta, size)

    # Perform assertions
    assert grid.shape == tuple(expected_shape), "Unexpected grid shape"
    np.testing.assert_allclose(grid.numpy(), expected_grid, rtol=1e-6, atol=1e-6)
    print("Test passed!")

# Run the test
test_create_affine_grid()

