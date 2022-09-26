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
    helpers.test_frontend_function(
        input_dtypes=["float32"],
        num_positional_args=7,
        as_variable_flags=[False],
        with_out=False,
        native_array_flags=[False],
        fw=fw,
        frontend="torch",
        fn_tree="conv2d",
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=3,
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
    helpers.test_frontend_function(
        input_dtypes=["float32"],
        num_positional_args=7,
        as_variable_flags=[False],
        with_out=False,
        native_array_flags=[False],
        fw=fw,
        frontend="torch",
        fn_tree="conv2d",
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
    )


@st.composite
def _int_or_tuple(draw, min_val, max_val):
    val = draw(
        random.choice(
            [
                st.integers(min_val, max_val),
                st.tuples(st.integers(min_val, max_val)),
                st.tuples(
                    st.integers(min_val, max_val),
                    st.integers(min_val, max_val),
                ),
            ]
        )
    )
    return val


@handle_cmd_line_args
@given(
    dtype_and_input_and_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=(1, 3, 6, 6),
    ),
    kernel_size=_int_or_tuple(2, 5),
    dilation=_int_or_tuple(1, 3),
    padding=_int_or_tuple(0, 2),
    stride=_int_or_tuple(1, 3),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.unfold"
    ),
)
def test_torch_unfold(
    dtype_and_input_and_shape,
    kernel_size,
    dilation,
    padding,
    stride,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    args_dtypes = list([dtype_and_input_and_shape[0]] + ["uint8"] * 4)
    helpers.test_frontend_function(
        input_dtypes=args_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.unfold",
        input=dtype_and_input_and_shape[1],
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


@handle_cmd_line_args
@given(
    dtype_and_input_and_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=(1, 12, 12),
    ),
    output_size=_int_or_tuple(3, 5),
    kernel_size=_int_or_tuple(2, 5),
    dilation=_int_or_tuple(1, 3),
    padding=_int_or_tuple(0, 2),
    stride=_int_or_tuple(1, 3),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.fold"
    ),
)
def test_torch_fold(
    dtype_and_input_and_shape,
    output_size,
    kernel_size,
    dilation,
    padding,
    stride,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    args_dtypes = list([dtype_and_input_and_shape[0]] + ["uint8"] * 5)
    helpers.test_frontend_function(
        input_dtypes=args_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.fold",
        input=dtype_and_input_and_shape[1],
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
