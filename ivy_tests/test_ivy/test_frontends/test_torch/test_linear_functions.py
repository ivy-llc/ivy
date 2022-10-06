# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def x_and_linear(draw, dtypes):
    dtype = draw(dtypes)
    in_features = draw(helpers.ints(min_value=1, max_value=2))
    out_features = draw(helpers.ints(min_value=1, max_value=2))

    x_shape = (
        1,
        1,
        in_features,
    )
    weight_shape = (out_features, in_features)
    bias_shape = (out_features,)

    x = draw(
        helpers.array_values(dtype=dtype[0], shape=x_shape, min_value=0, max_value=1)
    )
    weight = draw(
        helpers.array_values(
            dtype=dtype[0], shape=weight_shape, min_value=0, max_value=1
        )
    )
    bias = draw(
        helpers.array_values(dtype=dtype[0], shape=bias_shape, min_value=0, max_value=1)
    )
    return dtype, x, weight, bias


# linear
@handle_cmd_line_args
@given(
    dtype_x_weight_bias=x_and_linear(
        dtypes=helpers.get_dtypes("float", full=False),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.linear"
    ),
)
def test_linear(
    *,
    dtype_x_weight_bias,
    as_variable,
    num_positional_args,
    native_array,
):

    dtype, x, weight, bias = dtype_x_weight_bias
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.linear",
        rtol=1e-02,
        atol=1e-02,
        input=x,
        weight=weight,
        bias=bias,
    )
