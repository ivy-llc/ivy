# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _valid_istft(draw):
    # Generating a complex dtype and corresponding STFT matrix values
    dtype, stft_matrix = draw(
        helpers.dtype_and_values(
            available_dtypes=["complex64", "complex128"],
            max_value=65280,
            min_value=-65280,
            min_num_dims=2,  # STFT matrix usually has at least 2 dimensions
            min_dim_size=2,
            shared_dtype=True,
        )
    )
    # Randomly generating n_fft and hop_length
    n_fft = draw(helpers.ints(min_value=16, max_value=100))
    hop_length = draw(helpers.ints(min_value=1, max_value=50))

    # Return the generated parameters
    return dtype, stft_matrix, n_fft, hop_length


# --- Main --- #
# ------------ #


# Test function for istft
@handle_frontend_test(
    fn_tree="paddle.signal.istft",  # Assuming istft is under paddle.signal namespace
    dtype_x_and_args=_valid_istft(),
    test_with_out=st.just(False),
)
def test_paddle_istft(
    *,
    dtype_x_and_args,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, stft_matrix, n_fft, hop_length = dtype_x_and_args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        stft_matrix=stft_matrix[0],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=None,
        window=None,
        center=True,
        normalized=False,
        onesided=True,
        length=None,  # Optionally, you can add a strategy to generate this
        atol=1e-02,
        rtol=1e-02,
    )
