# global
import random
from hypothesis import given, assume, strategies as st

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


@st.composite
def int_or_tuple(draw):
    key_st = st.sampled_from(
        [
            st.integers(min_value=1),
            st.tuples(st.integers(min_value=1)),
            st.tuples(st.integers(min_value=1), st.integers(min_value=1)),
        ]
    )
    return draw(key_st)


@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=4,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.unfold"
    ),
)
def test_torch_unfold(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    x_dtype, x = dtype_and_input
    kernel_size = int_or_tuple()
    dilation = int_or_tuple()
    padding = int_or_tuple()
    stride = int_or_tuple()
    arg_dtypes = x_dtype + [int, int, int, int]
    if x.ndim() == 3:
        assume(x.shape()[0] and x.shape()[1] and x.shape()[2])
    elif x.ndim() == 4:
        assume(x.shape()[1] and x.shape()[2] and x.shape()[3])
    helpers.test_frontend_function(
        input_dtypes=arg_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.unfold",
        input=x,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
