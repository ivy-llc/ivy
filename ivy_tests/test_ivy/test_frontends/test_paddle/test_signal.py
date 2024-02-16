from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _valid_stft(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=["complex64", "complex128"],
            max_value=65280,
            min_value=-65280,
            min_num_dims=1,
            min_dim_size=2,
            shared_dtype=True,
        )
    )
    n_fft = draw(helpers.ints(min_value=16, max_value=100))
    hop_length = draw(helpers.ints(min_value=1, max_value=50))

    return dtype, x, n_fft, hop_length


# --- Main --- #
# ------------ #


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
        hop_length=None,
        win_length=None,
        window=None,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        atol=1e-02,
        rtol=1e-02,
    )
