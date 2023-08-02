# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import tensorflow as tf
import numpy as np


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


#inverse_mdct

#Generating MDCT COEFF
def generate_mdct_coefficients():
    num_samples = np.random.randint(2, 100)
    num_frequency_bins = np.random.randint(2, 128)
    return tf.constant(np.random.randn(num_samples, num_frequency_bins), dtype=tf.float32)



@st.composite
def valid_inverse_mdct(draw):
    input_dtype = draw(st.sampled_from([tf.float32]))
    mdct_coefficients = generate_mdct_coefficients()
    window_fn = draw(st.sampled_from([tf.signal.vorbis_window, tf.signal.hann_window]))
    norm = draw(st.one_of(st.none(), st.floats(min_value=0.1, max_value=1.0)))
    return [(input_dtype, mdct_coefficients, window_fn, norm)]

@handle_frontend_test(
    fn_tree="tensorflow.signal.inverse_mdct",
    dtype_x_and_args=valid_inverse_mdct(),
    test_with_out=st.just(False),
)
def test_tensorflow_inverse_mdct(
    *,
    dtype_x_and_args,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, mdct, window_fn, norm = dtype_x_and_args[0]
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        mdct=mdct,
        window_fn=window_fn,
        norm=norm,
        atol=1e-01,
    )
