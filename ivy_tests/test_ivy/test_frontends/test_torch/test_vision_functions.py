from hypothesis import given, assume, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# pixel_shuffle
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=1,
    ),
    factor=st.integers(min_value=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.pixel_shuffle"
    ),
)
def test_torch_pixel_shuffle(
    dtype_and_x,
    factor,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    input = x[0]
    assume(ivy.shape(input)[1] % (factor**2) == 0)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.pixel_shuffle",
        input=input,
        upscale_factor=factor,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=1,
    ),
    factor=st.integers(min_value=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.pixel_unshuffle"
    ),
)
def test_torch_pixel_unshuffle(
    dtype_and_x,
    factor,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    input = x[0]
    assume((ivy.shape(input)[2] % factor == 0) & (ivy.shape(input)[3] % factor == 0))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.pixel_unshuffle",
        input=input,
        downscale_factor=factor,
    )
