# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import tensorflow as tf
import numpy as np
from ivy.functional.frontends.tensorflow.signal import kaiser_bessel_derived_window

# kaiser_window
@handle_frontend_test(
    fn_tree="tensorflow.signal.kaiser_window",
    dtype_and_window_length=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer")
    ),
    dtype_and_beta=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("numeric"),
    test_with_out=st.just(False),
)
def test_tensorflow_kaiser_window(
    *,
    dtype_and_window_length,
    dtype_and_beta,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    window_length_dtype, window_length = dtype_and_window_length
    beta_dtype, beta = dtype_and_beta
    helpers.test_frontend_function(
        input_dtypes=[window_length_dtype[0], beta_dtype[0]],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        window_length=window_length,
        beta=beta,
        dtype=dtype,
    )


@st.composite
def valid_idct(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            max_value=65280,
            min_value=-65280,
            min_num_dims=1,
            min_dim_size=2,
            shared_dtype=True,
        )
    )
    n = None
    axis = -1
    norm = draw(st.sampled_from([None, "ortho"]))
    type = draw(helpers.ints(min_value=1, max_value=4))
    if norm == "ortho" and type == 1:
        norm = None
    return dtype, x, type, n, axis, norm


# idct
@handle_frontend_test(
    fn_tree="tensorflow.signal.idct",
    dtype_x_and_args=valid_idct(),
    test_with_out=st.just(False),
)
def test_tensorflow_idct(
    *,
    dtype_x_and_args,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, type, n, axis, norm = dtype_x_and_args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        atol=1e-01,
    )

def test_kaiser_bessel_derived_window_simple():
    # Define the window length and beta
    window_length = 10
    beta = 12.0

    # Call the function
    result = kaiser_bessel_derived_window(window_length, beta)

    # Check the result
    assert isinstance(result, tf.Tensor), "Result is not a TensorFlow tensor"
    assert result.shape == (window_length,), "Result has incorrect shape"
    assert np.all(np.isfinite(result.numpy())), "Result contains non-finite values"

@handle_frontend_test(
    fn_tree="tensorflow.signal.kaiser_bessel_derived_window",
    dtype_and_window_length=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer")
    ),
    dtype_and_beta=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("numeric"),
    test_with_out=st.just(False),
)
def test_tensorflow_kaiser_bessel_derived_window(
    *,
    dtype_and_window_length,
    dtype_and_beta,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    window_length_dtype, window_length = dtype_and_window_length
    beta_dtype, beta = dtype_and_beta
    helpers.test_frontend_function(
        input_dtypes=[window_length_dtype[0], beta_dtype[0]],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        window_length=window_length,
        beta=beta,
        dtype=dtype,
    )

