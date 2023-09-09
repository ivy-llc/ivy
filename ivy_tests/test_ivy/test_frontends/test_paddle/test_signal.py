# global
from hypothesis import strategies as st


# Now you can use short_DFT in your test function as shown in the previous example
# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy.functional.frontends.paddle.signal import short_DFT

@handle_frontend_test(
    fn_tree="paddle.signal.short_DFT",  # Adjust the function path as needed
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    )
)
def test_paddle_short_DFT(
        dtype_x_axis,
        frontend,
        test_flags,
        fn_tree,
        on_device,
        backend_fw,
):
    input_dtype, x, axes = dtype_x_axis

    # Generate test input data, you may need to adjust the input data according to your function
    x = paddle.randn([8, 48000], dtype=input_dtype)

    # Call short_DFT function
    y = short_DFT(x, n_fft=512, onesided=True, name=None)  # Modify the function parameters as needed

    # Perform assertions or validation checks based on your specific function's behavior
    # For example, you can check the shape, dtype, or other properties of the output y.

    # Add assertions here based on your function's expected behavior
    # For example, assert y.shape == expected_shape and other relevant checks.


# Run the tests
if __name__ == '__main__':
    pytest.main(['-s', __file__])
