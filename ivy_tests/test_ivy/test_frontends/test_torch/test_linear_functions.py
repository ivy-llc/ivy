# global
from hypothesis import given, strategies as st
import numpy as np
# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def x_and_bilinear(draw, dtypes):
    dtype = draw(dtypes)
    outer_batch_shape = draw(
        st.tuples(
            helpers.ints(min_value=3, max_value=5),
            helpers.ints(min_value=1, max_value=3),
            helpers.ints(min_value=1, max_value=3),
        )
    )
    inner_batch_shape = draw(
        st.tuples(
            helpers.ints(min_value=3, max_value=5),
            helpers.ints(min_value=1, max_value=3),
            helpers.ints(min_value=1, max_value=3),
        )
    )
    in1_features = draw(helpers.ints(min_value=1, max_value=3))
    in2_features = draw(helpers.ints(min_value=1, max_value=3))
    out_features = draw(helpers.ints(min_value=1, max_value=3))

    input1_shape = outer_batch_shape + inner_batch_shape + (in1_features,)
    input2_shape = outer_batch_shape + inner_batch_shape + (in2_features,)
    weight_shape = outer_batch_shape + (out_features,) + (in1_features,) + (in2_features,)
    bias_shape = outer_batch_shape + (out_features,)

    input1 = draw(
        helpers.array_values(dtype=dtype, shape=input1_shape, min_value=0, max_value=1)
    )
    input2 = draw(
        helpers.array_values(dtype=dtype, shape=input2_shape, min_value=0, max_value=1)
    )
    weight = draw(
        helpers.array_values(dtype=dtype, shape=weight_shape, min_value=0, max_value=1)
    )
    bias = draw(
        helpers.array_values(dtype=dtype, shape=bias_shape, min_value=0, max_value=1)
    )
    return dtype, input1, input2, weight, bias


@handle_cmd_line_args
@given(
    dtype_x_weight_bias=x_and_bilinear(
        dtypes=helpers.get_dtypes("float", full=False),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.bilinear"
    )
)
def test_torch_bilinear(
    dtype_input1_input2_weight_bias,
    weight,
    bias,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw
):
    dtype, input1, input2, weight, bias = dtype_input1_input2_weight_bias
    as_variable = [as_variable] * 4
    native_array = [native_array] * 4

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.bilinear",
        input1=np.asarray(input1, dtype=dtype),
        input2=np.asarray(input2, dtype=dtype),
        weight=np.asarray(weight, dtype=dtype),
        bias=np.asarray(bias, dtype=dtype),
    )
