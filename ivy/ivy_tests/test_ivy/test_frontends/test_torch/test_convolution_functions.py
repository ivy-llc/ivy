# global
import random
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy


@handle_cmd_line_args
@given(
    input=helpers.array_values(  # input
        dtype=ivy.float32, shape=(4, 3, 5, 5)  # (batch_size, d_in, h, w)
    ),
    weight=helpers.array_values(  # weight
        dtype=ivy.float32, shape=(3, 1, 3, 3)  # (d_out, d_in/groups, fh, fw)
    ),
    bias=helpers.array_values(dtype=ivy.float32, shape=3),
    stride=helpers.ints(min_value=1, max_value=3),  # stride
    dilation=helpers.ints(min_value=1, max_value=3),  # dilation
    padding=st.sampled_from([1, 2, 3, 4, 5, "same", "valid"]),
)
def test_torch_conv2d_1(input, weight, bias, stride, dilation, padding, fw):
    dtype = random.choice([ivy.float32, ivy.float64])
    groups = 3
    dtypes = [dtype] * 3
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        num_positional_args=7,
        as_variable_flags=False,
        with_out=False,
        native_array_flags=False,
        fw=fw,
        frontend="torch",
        fn_tree="conv2d",
        input=ivy.array(input, dtype=dtype),
        weight=ivy.array(weight, dtype=dtype),
        bias=ivy.array(bias, dtype=dtype),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@handle_cmd_line_args
@given(
    input=helpers.array_values(  # input
        dtype=ivy.float32, shape=(6, 3, 3, 4)  # (batch_size, d_in, h, w)
    ),
    weight=helpers.array_values(  # weight
        dtype=ivy.float32, shape=(6, 3, 2, 2)  # (d_out, d_in/groups, fh, fw)
    ),
    bias=helpers.array_values(dtype=ivy.float32, shape=6),
    stride=helpers.ints(min_value=1, max_value=3),  # stride
    dilation=helpers.ints(min_value=1, max_value=3),  # dilation
    padding=st.sampled_from([1, 2, 3, 4, 5, "same", "valid"]),
)
def test_torch_conv2d_2(input, weight, bias, stride, dilation, padding, fw):
    dtype = random.choice([ivy.float32, ivy.float64])
    groups = 1
    dtypes = [dtype] * 3
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        num_positional_args=7,
        as_variable_flags=False,
        with_out=False,
        native_array_flags=False,
        fw=fw,
        frontend="torch",
        fn_tree="conv2d",
        input=ivy.array(input, dtype=dtype),
        weight=ivy.array(weight, dtype=dtype),
        bias=ivy.array(bias, dtype=dtype),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
