from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Updated test function
@handle_frontend_test(
    fn_tree="paddle.signal.stft",
    dtype_x_and_args=_valid_stft(),
    test_with_out=st.just(False),
)
def test_paddle_stft(
    *,
    dtype_x_and_args,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, n_fft, hop_length = dtype_x_and_args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=None,
        window=None,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        atol=1e-02,
        rtol=1e-02,
    )
