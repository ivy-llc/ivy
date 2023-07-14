# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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
@handle_frontend_test(
    fn_tree="torch.nn.functional.linear",
    dtype_x_weight_bias=x_and_linear(
        dtypes=helpers.get_dtypes("float", full=False),
    ),
)
def test_torch_linear(
    *,
    dtype_x_weight_bias,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, weight, bias = dtype_x_weight_bias
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        weight=weight,
        bias=bias,
    )
