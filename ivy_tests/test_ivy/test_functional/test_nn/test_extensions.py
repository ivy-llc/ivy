"""Tests for extension FFT."""

# global
from hypothesis import given, strategies as st, assume

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


min_fft_points = 2


@st.composite
def x_and_fft(draw, dtypes):
    dtype = draw(dtypes)
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2,
            max_dim_size=100,
            min_num_dims=1,
            max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype = dtype[0],
            shape=tuple(x_dim),
        )
    )
    dim = draw(
        helpers.get_axis(
            shape=x_dim,
            allow_neg=True,
            allow_none=False,
            max_size=1
        )
    )
    norm = draw(st.sampled_from(["backward","forward","ortho"]))
    n = draw(st.integers(min_fft_points,256))
    return dtype,x,dim,norm,n


@handle_cmd_line_args
@given(
    d_x_d_n_n=x_and_fft(
        helpers.get_dtypes("complex")
    ),
    num_positional_args=helpers.num_positional_args(fn_name="fft"),
)
def test_fft(
    *,
    d_x_d_n_n,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, dim,norm,n = d_x_d_n_n
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="fft",
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=False,
        ground_truth_backend="numpy",
        x = x,
        dim = dim,
        norm = norm,
        n = n
    )