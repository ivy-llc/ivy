# global
import numpy as np
from hypothesis import assume, given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(
                    draw(helpers.get_dtypes("float")),
                ),
                length=1,
            ),
            key="dtype",
        )
    )


@st.composite
def _generate_prelu_arrays(draw):
    arr_size = draw(helpers.ints(min_value=2, max_value=5))

    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    input = draw(
        helpers.array_values(dtype=dtype, shape=(arr_size), min_value=0, max_value=10)
    )
    weight = draw(
        helpers.array_values(dtype=dtype, shape=(1,), min_value=0, max_value=1.0)
    )
    input_weight = input, weight
    return dtype, input_weight


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.sigmoid"
    ),
)
def test_torch_sigmoid(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.sigmoid",
        input=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.softmax"
    ),
)
def test_torch_softmax(
    dtype_x_and_axis,
    as_variable,
    with_out,
    dtypes,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.softmax",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        dtype=dtypes[0],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.gelu"
    ),
)
def test_torch_gelu(
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.gelu",
        input=np.asarray(x, dtype=input_dtype),
        rtol=1e-02,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.leaky_relu"
    ),
    alpha=st.floats(min_value=0, max_value=1),
)
def test_torch_leaky_relu(
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    fw,
    alpha,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.leaky_relu",
        input=np.asarray(x, dtype=input_dtype),
        negative_slope=alpha,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.tanh"
    ),
)
def test_torch_tanh(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.tanh",
        input=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.logsigmoid"
    ),
)
def test_torch_logsigmoid(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.logsigmoid",
        input=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.softmin"
    ),
)
def test_torch_softmin(
    dtype_x_and_axis,
    as_variable,
    with_out,
    dtypes,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_x_and_axis
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.softmin",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        dtype=dtypes[0],
    )


# threshold
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.threshold"
    ),
)
def test_torch_threshold(
    dtype_and_input,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.threshold",
        input=np.asarray(input, dtype=input_dtype),
        threshold=0.5,
        value=20,
        inplace=False,
    )


# threshold_
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.threshold_"
    ),
)
def test_torch_threshold_(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.threshold_",
        input=np.asarray(input, dtype=input_dtype),
        threshold=0.5,
        value=20,
    )


# relu6
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.relu6"
    ),
)
def test_torch_relu6(
    dtype_and_input,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.relu6",
        input=np.asarray(input, dtype=input_dtype),
        inplace=False,
    )


# elu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.elu"
    ),
    alpha=helpers.floats(min_value=0.1, max_value=1.0, exclude_min=True),
)
def test_torch_elu(
    dtype_and_input,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.elu",
        input=np.asarray(input, dtype=input_dtype),
        alpha=alpha,
        inplace=False,
    )


# ToDo test for values once inplace test implemented
# elu_
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.elu_"
    ),
    alpha=helpers.floats(min_value=0, max_value=1, exclude_min=True),
)
def test_torch_elu_(
    dtype_and_input,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.elu_",
        input=np.asarray(input, dtype=input_dtype),
        alpha=alpha,
        test_values=False,
    )


# celu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.celu"
    ),
    alpha=helpers.floats(min_value=0.1, max_value=1.0, exclude_min=True),
)
def test_torch_celu(
    dtype_and_input,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.celu",
        input=np.asarray(input, dtype=input_dtype),
        alpha=alpha,
        inplace=False,
    )


# selu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.selu"
    ),
)
def test_torch_selu(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.selu",
        input=np.asarray(input, dtype=input_dtype),
        inplace=False,
    )


# prelu
@handle_cmd_line_args
@given(
    dtype_input_and_weight=_generate_prelu_arrays(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.prelu"
    ),
)
def test_torch_prelu(
    dtype_input_and_weight,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    dtype, inputs = dtype_input_and_weight
    input, weight = inputs
    assume("float16" not in dtype)
    helpers.test_frontend_function(
        input_dtypes=[dtype, dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.prelu",
        input=np.asarray(input, dtype=dtype),
        weight=np.asarray(weight, dtype=dtype),
    )


# rrelu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.rrelu"
    ),
    lower=helpers.floats(min_value=0, max_value=0.5, exclude_min=True),
    upper=helpers.floats(min_value=0.5, max_value=1.0, exclude_min=True),
)
def test_torch_rrelu(
    dtype_and_input,
    lower,
    upper,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.rrelu",
        input=np.asarray(input, dtype=input_dtype),
        lower=lower,
        upper=upper,
        inplace=False,
    )


# ToDo test for values once inplace test implemented
# rrelu_
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.rrelu_"
    ),
    lower=helpers.floats(min_value=0, max_value=0.5, exclude_min=True),
    upper=helpers.floats(min_value=0.5, max_value=1.0, exclude_min=True),
)
def test_torch_rrelu_(
    dtype_and_input,
    lower,
    upper,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.rrelu_",
        input=np.asarray(input, dtype=input_dtype),
        lower=lower,
        upper=upper,
        test_values=False,
    )


# hardshrink
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.hardshrink"
    ),
    lambd=helpers.floats(min_value=0, max_value=1, exclude_min=True),
)
def test_torch_hardshrink(
    dtype_and_input,
    lambd,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.hardshrink",
        input=np.asarray(input, dtype=input_dtype),
        lambd=lambd,
    )


# softsign
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.softsign"
    ),
)
def test_torch_softsign(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.softsign",
        input=np.asarray(input, dtype=input_dtype),
    )


# softshrink
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.softshrink"
    ),
    lambd=helpers.floats(min_value=0, max_value=1, exclude_min=True),
)
def test_torch_softshrink(
    dtype_and_input,
    lambd,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.softshrink",
        input=np.asarray(input, dtype=input_dtype),
        lambd=lambd,
    )


# silu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.silu"
    ),
)
def test_torch_silu(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.silu",
        input=np.asarray(input, dtype=input_dtype),
        inplace=False,
    )


@st.composite
def _glu_arrays(draw):
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    shape = draw(st.shared(helpers.ints(min_value=1, max_value=5)))
    shape = shape * 2
    input = draw(helpers.array_values(dtype=dtype, shape=(shape, shape)))
    dim = draw(st.shared(helpers.get_axis(shape=(shape, shape), force_int=True)))
    return dtype, input, dim


# glu
@handle_cmd_line_args
@given(
    dtype_input_dim=_glu_arrays(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.glu"
    ),
)
def test_torch_glu(
    dtype_input_dim,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input, dim = dtype_input_dim
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.glu",
        input=np.asarray(input, dtype=input_dtype),
        dim=dim,
    )


# log_softmax
@handle_cmd_line_args
@given(
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.log_softmax"
    ),
)
def test_torch_log_softmax(
    dtype_x_and_axis,
    as_variable,
    with_out,
    dtypes,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.log_softmax",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        dtype=dtypes[0],
    )


# tanhshrink
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.tanhshrink"
    ),
)
def test_torch_tanhshrink(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.tanhshrink",
        input=np.asarray(input, dtype=input_dtype),
    )


# leaky_relu_
# ToDo test for value test once inplace testing implemented
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.leaky_relu_"
    ),
    alpha=st.floats(min_value=0, max_value=1, exclude_min=True),
)
def test_torch_leaky_relu_(
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    fw,
    alpha,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.leaky_relu_",
        input=np.asarray(x, dtype=input_dtype),
        negative_slope=alpha,
        test_values=False,
    )


# hardswish
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.hardswish"
    ),
)
def test_torch_hardswish(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nn.functional.hardswish",
        input=np.asarray(input, dtype=input_dtype),
        inplace=False,
    )
