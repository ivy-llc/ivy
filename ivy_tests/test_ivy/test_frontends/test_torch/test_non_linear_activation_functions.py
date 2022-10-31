# global
import ivy
from hypothesis import assume, given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _generate_prelu_arrays(draw):
    arr_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    input = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(arr_size), min_value=0, max_value=10
        )
    )
    weight = draw(
        helpers.array_values(dtype=dtype[0], shape=(1,), min_value=0, max_value=1.0)
    )
    input_weight = input, weight
    return dtype, input_weight


def _filter_dtypes(input_dtype):
    assume(("bfloat16" not in input_dtype) and ("float16" not in input_dtype))


# sigmoid
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.sigmoid"
    ),
)
def test_torch_sigmoid(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.sigmoid",
        atol=1e-2,
        input=x[0],
    )


# softmax
@handle_cmd_line_args
@given(
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtypes=helpers.get_dtypes("float", none=True, full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.softmax"
    ),
)
def test_torch_softmax(
    dtype_x_and_axis,
    as_variable,
    dtypes,
    num_positional_args,
    native_array,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.softmax",
        input=x[0],
        dim=axis,
        _stacklevel=3,
        dtype=ivy.as_ivy_dtype(dtypes[0]),
    )


# gelu
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.gelu"
    ),
)
def test_torch_gelu(
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.gelu",
        rtol=1e-02,
        input=x[0],
    )


# leaky_relu
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.leaky_relu"
    ),
    alpha=st.floats(min_value=0.0, max_value=1.0),
    with_inplace=st.booleans(),
)
def test_torch_leaky_relu(
    dtype_and_x,
    with_inplace,
    num_positional_args,
    as_variable,
    native_array,
    alpha,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.leaky_relu",
        input=x[0],
        negative_slope=alpha,
    )


# tanh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.tanh"
    ),
)
def test_torch_tanh(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.tanh",
        atol=1e-2,
        input=x[0],
    )


# logsigmoid
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.logsigmoid"
    ),
)
def test_torch_logsigmoid(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.logsigmoid",
        input=x[0],
    )


# softmin
@handle_cmd_line_args
@given(
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtypes=helpers.get_dtypes("float", none=True, full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.softmin"
    ),
)
def test_torch_softmin(
    dtype_x_and_axis,
    as_variable,
    dtypes,
    num_positional_args,
    native_array,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.softmin",
        input=x[0],
        dim=axis,
        dtype=ivy.as_ivy_dtype(dtypes[0]),
    )


# threshold
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.threshold"
    ),
    threshold=helpers.floats(min_value=0.0, max_value=1.0),
    value=helpers.ints(min_value=5, max_value=20),
    with_inplace=st.booleans(),
)
def test_torch_threshold(
    dtype_and_input,
    threshold,
    value,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.threshold",
        input=input[0],
        threshold=threshold,
        value=value,
    )


# threshold_
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.threshold_"
    ),
    threshold=helpers.floats(min_value=0.0, max_value=1.0),
    value=helpers.ints(min_value=5, max_value=20),
    with_inplace=st.booleans(),
)
def test_torch_threshold_(
    dtype_and_input,
    threshold,
    value,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.threshold_",
        input=input[0],
        threshold=threshold,
        value=value,
    )


# relu6
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.relu6"
    ),
    with_inplace=st.booleans(),
)
def test_torch_relu6(
    dtype_and_input,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.relu6",
        input=input[0],
    )


# elu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.elu"
    ),
    alpha=helpers.floats(min_value=0.1, max_value=1.0, exclude_min=True),
    with_inplace=st.booleans(),
)
def test_torch_elu(
    dtype_and_input,
    alpha,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.elu",
        input=input[0],
        alpha=alpha,
    )


# elu_
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.elu_"
    ),
    alpha=helpers.floats(min_value=0, max_value=1, exclude_min=True),
    with_inplace=st.booleans(),
)
def test_torch_elu_(
    dtype_and_input,
    alpha,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.elu_",
        test_values=False,
        input=input[0],
        alpha=alpha,
    )


# celu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.celu"
    ),
    alpha=helpers.floats(min_value=0.1, max_value=1.0, exclude_min=True),
    with_inplace=st.booleans(),
)
def test_torch_celu(
    dtype_and_input,
    alpha,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.celu",
        input=input[0],
        alpha=alpha,
    )


# mish
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.mish"
    ),
    with_inplace=st.booleans(),
)
def test_torch_mish(
    dtype_and_input,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.mish",
        input=input[0],
    )


# selu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.selu"
    ),
    with_inplace=st.booleans(),
)
def test_torch_selu(
    dtype_and_input,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.selu",
        input=input[0],
    )


# prelu
@handle_cmd_line_args
@given(
    dtype_input_and_weight=_generate_prelu_arrays(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.prelu"
    ),
)
def test_torch_prelu(
    dtype_input_and_weight,
    as_variable,
    num_positional_args,
    native_array,
):
    dtype, inputs = dtype_input_and_weight
    _filter_dtypes(dtype)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.prelu",
        input=inputs[0],
        weight=inputs[1],
    )


# rrelu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.rrelu"
    ),
    lower=helpers.floats(min_value=0, max_value=0.5, exclude_min=True),
    upper=helpers.floats(min_value=0.5, max_value=1.0, exclude_min=True),
    with_inplace=st.booleans(),
)
def test_torch_rrelu(
    dtype_and_input,
    lower,
    upper,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.rrelu",
        input=input[0],
        lower=lower,
        upper=upper,
    )


# rrelu_
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.rrelu_"
    ),
    lower=helpers.floats(min_value=0, max_value=0.5, exclude_min=True),
    upper=helpers.floats(min_value=0.5, max_value=1.0, exclude_min=True),
    with_inplace=st.booleans(),
)
def test_torch_rrelu_(
    dtype_and_input,
    lower,
    upper,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.rrelu_",
        test_values=False,
        input=input[0],
        lower=lower,
        upper=upper,
    )


# hardshrink
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.hardshrink"
    ),
    lambd=helpers.floats(min_value=0, max_value=1, exclude_min=True),
)
def test_torch_hardshrink(
    dtype_and_input,
    lambd,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.hardshrink",
        input=input[0],
        lambd=lambd,
    )


# softsign
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.softsign"
    ),
)
def test_torch_softsign(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.softsign",
        input=input[0],
    )


# softshrink
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.softshrink"
    ),
    lambd=helpers.floats(min_value=0, max_value=1, exclude_min=True),
)
def test_torch_softshrink(
    dtype_and_input,
    lambd,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.softshrink",
        input=input[0],
        lambd=lambd,
    )


# silu
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.silu"
    ),
    with_inplace=st.booleans(),
)
def test_torch_silu(
    dtype_and_input,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.silu",
        rtol=1e-2,
        atol=1e-2,
        input=input[0],
    )


@st.composite
def _glu_arrays(draw):
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    shape = draw(st.shared(helpers.ints(min_value=1, max_value=5)))
    shape = shape * 2
    input = draw(helpers.array_values(dtype=dtype[0], shape=(shape, shape)))
    dim = draw(st.shared(helpers.get_axis(shape=(shape,), force_int=True)))
    return dtype, input, dim


# glu
@handle_cmd_line_args
@given(
    dtype_input_dim=_glu_arrays(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.glu"
    ),
)
def test_torch_glu(
    dtype_input_dim,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input, dim = dtype_input_dim
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.glu",
        input=input[0],
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
    dtypes=helpers.get_dtypes("float", none=False, full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.log_softmax"
    ),
)
def test_torch_log_softmax(
    dtype_x_and_axis,
    as_variable,
    dtypes,
    num_positional_args,
    native_array,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.log_softmax",
        input=x[0],
        dim=axis,
        _stacklevel=3,
        dtype=ivy.as_ivy_dtype(dtypes[0]),
    )


# tanhshrink
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.tanhshrink"
    ),
)
def test_torch_tanhshrink(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.tanhshrink",
        input=input[0],
    )


# leaky_relu_
# ToDo test for value test once inplace testing implemented
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.leaky_relu_"
    ),
    alpha=st.floats(min_value=0, max_value=1, exclude_min=True),
    with_inplace=st.booleans(),
)
def test_torch_leaky_relu_(
    dtype_and_x,
    alpha,
    with_inplace,
    num_positional_args,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.leaky_relu_",
        test_values=False,
        input=x[0],
        negative_slope=alpha,
    )


# hardswish
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        safety_factor_scale="log",
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.hardswish"
    ),
    with_inplace=st.booleans(),
)
def test_torch_hardswish(
    dtype_and_input,
    with_inplace,
    num_positional_args,
    as_variable,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.hardswish",
        input=input[0],
    )


# hardsigmoid
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.hardsigmoid"
    ),
    with_inplace=st.booleans(),
)
def test_torch_hardsigmoid(
    dtype_and_input,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.hardsigmoid",
        input=input[0],
    )


# hardtanh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.hardtanh"
    ),
    max_val=st.floats(min_value=0, max_value=1, exclude_min=True),
    with_inplace=st.booleans(),
)
def test_torch_hardtanh(
    dtype_and_x,
    max_val,
    with_inplace,
    num_positional_args,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    max_min = max_val, -max_val
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.hardtanh",
        input=x[0],
        min_val=max_min[1],
        max_val=max_min[0],
    )


# hardtanh_
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.hardtanh_"
    ),
    max_val=st.floats(min_value=0, max_value=1, exclude_min=True),
    with_inplace=st.booleans(),
)
def test_torch_hardtanh_(
    dtype_and_x,
    max_val,
    with_inplace,
    num_positional_args,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    max_min = max_val, -max_val
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.hardtanh_",
        test_values=False,
        input=x[0],
        min_val=max_min[1],
        max_val=max_min[0],
    )


# normalize
@handle_cmd_line_args
@given(
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.normalize"
    ),
    p=helpers.ints(min_value=2, max_value=5),
    without=st.booleans(),
)
def test_torch_normalize(
    dtype_x_and_axis,
    p,
    without,
    num_positional_args,
    as_variable,
    native_array,
):
    dtype, x, axis = dtype_x_and_axis
    _filter_dtypes(dtype)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=without,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.normalize",
        input=x[0],
        p=p,
        dim=axis,
        eps=1e-12,
    )


@st.composite
def _generate_data_layer_norm(
    draw,
    *,
    available_dtypes,
    large_abs_safety_factor=40,
    small_abs_safety_factor=40,
    safety_factor_scale="log",
    min_num_dims=1,
    max_num_dims=5,
    valid_axis=True,
    allow_neg_axes=False,
    max_axes_size=1,
    force_int_axis=True,
    ret_shape=True,
    abs_smallest_val=None,
    allow_inf=False,
    allow_nan=False,
    exclude_min=False,
    exclude_max=False,
    min_value=None,
    max_value=None,
    shared_dtype=False,
):
    results = draw(
        helpers.dtype_values_axis(
            available_dtypes=available_dtypes,
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale=safety_factor_scale,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            valid_axis=valid_axis,
            allow_neg_axes=allow_neg_axes,
            max_axes_size=max_axes_size,
            force_int_axis=force_int_axis,
            ret_shape=ret_shape,
        )
    )

    dtype, values, axis, shape = results

    weight_shape = shape[axis:]
    bias_shape = shape[axis:]

    arg_dict = {
        "available_dtypes": dtype,
        "abs_smallest_val": abs_smallest_val,
        "min_value": min_value,
        "max_value": max_value,
        "large_abs_safety_factor": large_abs_safety_factor,
        "small_abs_safety_factor": small_abs_safety_factor,
        "allow_inf": allow_inf,
        "allow_nan": allow_nan,
        "exclude_min": exclude_min,
        "exclude_max": exclude_max,
        "min_num_dims": min_num_dims,
        "max_num_dims": max_num_dims,
        "shared_dtype": shared_dtype,
        "ret_shape": False,
    }

    results_weight = draw(helpers.dtype_and_values(shape=weight_shape, **arg_dict))
    results_bias = draw(helpers.dtype_and_values(shape=bias_shape, **arg_dict))
    results_new_std = draw(helpers.dtype_and_values(shape=shape, **arg_dict))

    _, weight_values = results_weight
    _, bias_values = results_bias
    _, new_std_values = results_new_std

    axis = shape[axis:]

    return dtype, values, axis, weight_values, bias_values, new_std_values


@handle_cmd_line_args
@given(
    dtype_x_and_axis=_generate_data_layer_norm(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.layer_norm"
    ),
)
def test_torch_layer_norm(
    dtype_x_and_axis,
    num_positional_args,
    as_variable,
    native_array,
):
    dtype, x, axis, weight, bias, new_std = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.layer_norm",
        rtol=1e-1,
        atol=1e-1,
        input=x[0],
        normalized_shape=axis,
        weight=weight[0],
        bias=bias[0],
        eps=1e-12,
    )
