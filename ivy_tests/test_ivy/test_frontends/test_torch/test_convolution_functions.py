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
def _int_or_tuple(draw):
    val = draw(
        random.choice(
            [
                st.integers(min_value=0, max_value=255),
                st.tuples(st.integers(min_value=0, max_value=255)),
                st.tuples(
                    st.integers(min_value=0, max_value=255),
                    st.integers(min_value=0, max_value=255),
                ),
            ]
        )
    )
    return val


@handle_cmd_line_args
@given(
    dtype_and_input_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=4,
        min_dim_size=0,
        ret_shape=True,
    ),
    kernel_size=_int_or_tuple(),
    dilation=_int_or_tuple(),
    padding=_int_or_tuple(),
    stride=_int_or_tuple(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.unfold"
    ),
)
def test_torch_unfold(
    dtype_and_input_shape,
    kernel_size,
    dilation,
    padding,
    stride,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    args_dtypes = list([dtype_and_input_shape[0]] + ["uint8"] * 4)
    input_shape = dtype_and_input_shape[2]
    input_ndims = len(input_shape)
    if input_ndims == 3:
        assume(input_shape[0] and input_shape[1] and input_shape[2])
    elif input_ndims == 4:
        assume(input_shape[1] and input_shape[2] and input_shape[3])
    helpers.test_frontend_function(
        input_dtypes=args_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.unfold",
        input=dtype_and_input_shape[1],
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
