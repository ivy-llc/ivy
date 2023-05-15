# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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
    on_device,
):
    window_length_dtype, window_length = dtype_and_window_length
    beta_dtype, beta = dtype_and_beta
    helpers.test_frontend_function(
        input_dtypes=[window_length_dtype[0], beta_dtype[0]],
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
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=10,
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
    on_device,
):
    input_dtype, x, type, n, axis, norm = dtype_x_and_args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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


# stft
@st.composite
def valid_stft_params(draw):
    # draw data types
    dtype = draw(helpers.get_dtypes("numeric"))
    # Draw values for the input signal x
    x = draw(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=2, max_size=1000)
    )
    # Draw values for the window function size
    frame_length = draw(st.integers(min_value=2, max_value=1000))
    # Draw values for the hop size between adjacent frames
    frame_step = draw(st.integers(min_value=1, max_value=frame_length))
    # Draw values for the window function type
    window_fn = draw(
        st.sampled_from(["hann", "hamming", "rectangle", "blackman", "bartlett"])
    )
    # Draw values for the FFT size
    fft_length = draw(st.integers(min_value=frame_length, max_value=frame_length * 4))
    return dtype, x, frame_length, frame_step, window_fn, fft_length


@handle_frontend_test(
    fn_tree="tensorflow.signal.stft",
    dtype_x_and_args=valid_stft_params(),
    test_with_out=st.just(False),
)
def test_tensorflow_stft(
    *,
    dtype_x_and_args,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, frame_length, frame_step, window_fn, fft_length = dtype_x_and_args
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        signals=x[0],
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=window_fn,
        fft_length=fft_length,
        pad_end=True,
        name=None,
    )
