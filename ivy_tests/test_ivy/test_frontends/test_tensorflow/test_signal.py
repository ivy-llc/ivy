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
def stft_st(draw):
    dtype_s, signal = draw(helpers.dtype_and_values(dtype=["float32"],
                                                    min_num_dims=1))
    dtype_fft, fft_l = draw(helpers.dtype_and_values(dtype=["int32"],
                                                     min_value=1, shape=(1,),
                                                     large_abs_safety_factor=1,
                                                     small_abs_safety_factor=1))
    dtype_fl, fl = draw(helpers.dtype_and_values(dtype=["int32"],
                                                 min_value=fft_l[0][0], shape=(1,),
                                                 large_abs_safety_factor=1,
                                                 small_abs_safety_factor=1))
    dtype_fs, fs = draw(helpers.dtype_and_values(dtype=["int32"], min_value=1,
                                                 shape=(1,),
                                                 large_abs_safety_factor=1,
                                                 small_abs_safety_factor=1))
    dtypes = [dtype_s[0], dtype_fl[0], dtype_fs[0], dtype_fft[0]]
    return dtypes, signal, fl[0][0], fs[0][0], fft_l[0][0]


@handle_frontend_test(
    fn_tree="tensorflow.signal.stft",
    params=stft_st()
)
def test_stft(
    *,
    params,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtypes, signal, frame_length, frame_step, fft_length = params
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        signal=signal,
        frame_length=fl,
        frame_step=fs,
        fft_length=fft_l
    )
