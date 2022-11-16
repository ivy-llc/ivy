import ivy_tests.test_ivy.helpers as helpers
from hypothesis import given, strategies as st
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    # generate pereodic boolean
    periodic=st.booleans(),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.signal.hann_window"
    ),
)
def test_tensorflow_hann_window(
    periodic,
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,

):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="signal.hann_window",
        window_length=x[0],
        periodic=periodic,
        dtype=dtype[0],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),

    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.signal.inverse_stft"
    ),
)
# inverse_stft function
def test_tensorflow_inverse_stft(
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,

):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="signal.inverse_stft",
        stfts=x[0],
        frame_length=x[1],
        frame_step=x[2],
        fft_length=x[3],
        window_fn=x[4],
        dtype=dtype[0],
    )
