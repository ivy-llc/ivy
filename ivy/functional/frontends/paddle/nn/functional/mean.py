import ivy
from ivy.func_wrapper import to_ivy

@to_ivy
def mean_paddle(x):
    return ivy.mean(x)




def test_mean_paddle():
    # Create an example input tensor
    input_tensor = ivy.array([1.0, 2.0, 3.0, 4.0])

    # Calculate the mean using the mean_paddle function
    mean_result = mean_paddle(input_tensor)

    # Calculate the mean using the numpy backend for comparison
    numpy_mean = ivy.to_numpy(ivy.mean(ivy.to_numpy(input_tensor)))

    # Check if the calculated mean matches the numpy_mean
    assert ivy.to_numpy(mean_result) == numpy_mean

if __name__ == "__main__":
    test_mean_paddle()