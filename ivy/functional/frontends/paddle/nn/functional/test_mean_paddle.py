import ivy
import paddle
import ivy_tests.helpers as helpers

def test_mean():
    # Create a PaddlePaddle tensor
    x = ivy.array([1, 2, 3, 4, 5], 'float32', dev_str='cpu')

    # Calculate the mean using the custom function
    custom_mean = ivy.mean(x)

    # Calculate the mean using PaddlePaddle's built-in function
    paddle_mean = paddle.nn.mean(x)

    # Check if the custom mean matches the built-in mean
    assert ivy.array_equal(custom_mean, paddle_mean)


