# global
import ivy
from hypothesis import assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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
@handle_frontend_test(
    fn_tree="torch.nn.functional.sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_sigmoid(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-2,
        input=x[0],
    )


# softmax
@handle_frontend_test(
    fn_tree="torch.nn.functional.softmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtypes=helpers.get_dtypes("float", none=True, full=False),
)
def test_torch_softmax(
    *,
    dtype_x_and_axis,
    dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        _stacklevel=3,
        dtype=ivy.as_ivy_dtype(dtypes[0]),
    )


# gelu
@handle_frontend_test(
    fn_tree="torch.nn.functional.gelu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_gelu(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        input=x[0],
    )


# leaky_relu
@handle_frontend_test(
    fn_tree="torch.nn.functional.leaky_relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    alpha=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
    test_inplace=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_leaky_relu(
    *,
    dtype_and_x,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        rtol=1e-02,
        atol=1e-02,
        negative_slope=alpha,
    )


# tanh
@handle_frontend_test(
    fn_tree="torch.nn.functional.tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_tanh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-2,
        input=x[0],
    )


# logsigmoid
@handle_frontend_test(
    fn_tree="torch.nn.functional.logsigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_logsigmoid(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


# softmin
@handle_frontend_test(
    fn_tree="torch.nn.functional.softmin",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtypes=helpers.get_dtypes("float", none=True, full=False),
)
def test_torch_softmin(
    *,
    dtype_x_and_axis,
    dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        dtype=ivy.as_ivy_dtype(dtypes[0]),
    )


# threshold
@handle_frontend_test(
    fn_tree="torch.nn.functional.threshold",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    threshold=helpers.floats(min_value=0.0, max_value=1.0),
    value=helpers.ints(min_value=5, max_value=20),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_threshold(
    *,
    dtype_and_input,
    threshold,
    value,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        threshold=threshold,
        value=value,
    )


# threshold_
@handle_frontend_test(
    fn_tree="torch.nn.functional.threshold_",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    threshold=helpers.floats(min_value=0.0, max_value=1.0),
    value=helpers.ints(min_value=5, max_value=20),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_threshold_(
    *,
    dtype_and_input,
    threshold,
    value,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        threshold=threshold,
        value=value,
    )


# relu6
@handle_frontend_test(
    fn_tree="torch.nn.functional.relu6",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_inplace=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_relu6(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# elu
@handle_frontend_test(
    fn_tree="torch.nn.functional.elu",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    alpha=helpers.floats(min_value=0.1, max_value=1.0, exclude_min=True),
    test_inplace=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_elu(
    *,
    dtype_and_input,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        alpha=alpha,
    )


# elu_
@handle_frontend_test(
    fn_tree="torch.nn.functional.elu_",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    alpha=helpers.floats(min_value=0, max_value=1, exclude_min=True),
    test_inplace=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_elu_(
    *,
    dtype_and_input,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        input=input[0],
        alpha=alpha,
    )


# celu
@handle_frontend_test(
    fn_tree="torch.nn.functional.celu",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    alpha=helpers.floats(min_value=0.1, max_value=1.0, exclude_min=True),
    test_inplace=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_celu(
    *,
    dtype_and_input,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        alpha=alpha,
    )


# mish
@handle_frontend_test(
    fn_tree="torch.nn.functional.mish",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_mish(
    *,
    dtype_and_input,
    fn_tree,
    frontend,
    on_device,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# relu
@handle_frontend_test(
    fn_tree="torch.nn.functional.relu",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_relu(
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=input[0],
    )


# relu_
@handle_frontend_test(
    fn_tree="torch.nn.functional.relu_",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_torch_relu_(
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=input[0],
    )


# selu
@handle_frontend_test(
    fn_tree="torch.nn.functional.selu",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_selu(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# prelu
@handle_frontend_test(
    fn_tree="torch.nn.functional.prelu",
    dtype_input_and_weight=_generate_prelu_arrays(),
)
def test_torch_prelu(
    *,
    dtype_input_and_weight,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, inputs = dtype_input_and_weight
    _filter_dtypes(dtype)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        weight=inputs[1],
    )


# rrelu
@handle_frontend_test(
    fn_tree="torch.nn.functional.rrelu",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    lower=helpers.floats(min_value=0, max_value=0.5, exclude_min=True),
    upper=helpers.floats(min_value=0.5, max_value=1.0, exclude_min=True),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_rrelu(
    *,
    dtype_and_input,
    lower,
    upper,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        lower=lower,
        upper=upper,
    )


# rrelu_
@handle_frontend_test(
    fn_tree="torch.nn.functional.rrelu_",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    lower=helpers.floats(min_value=0, max_value=0.5, exclude_min=True),
    upper=helpers.floats(min_value=0.5, max_value=1.0, exclude_min=True),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_rrelu_(
    *,
    dtype_and_input,
    lower,
    upper,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        input=input[0],
        lower=lower,
        upper=upper,
    )


# hardshrink
@handle_frontend_test(
    fn_tree="torch.nn.functional.hardshrink",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    lambd=helpers.floats(min_value=0, max_value=1, exclude_min=True),
)
def test_torch_hardshrink(
    *,
    dtype_and_input,
    lambd,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        lambd=lambd,
    )


# softsign
@handle_frontend_test(
    fn_tree="torch.nn.functional.softsign",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_softsign(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# softshrink
@handle_frontend_test(
    fn_tree="torch.nn.functional.softshrink",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    lambd=helpers.floats(min_value=0, max_value=1, exclude_min=True),
)
def test_torch_softshrink(
    *,
    dtype_and_input,
    lambd,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        lambd=lambd,
    )


# silu
@handle_frontend_test(
    fn_tree="torch.nn.functional.silu",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_silu(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
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
@handle_frontend_test(
    fn_tree="torch.nn.functional.glu",
    dtype_input_dim=_glu_arrays(),
)
def test_torch_glu(
    *,
    dtype_input_dim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input, dim = dtype_input_dim
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dim=dim,
    )


# log_softmax
@handle_frontend_test(
    fn_tree="torch.nn.functional.log_softmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtypes=helpers.get_dtypes("float", none=False, full=False),
)
def test_torch_log_softmax(
    *,
    dtype_x_and_axis,
    dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        _stacklevel=3,
        dtype=ivy.as_ivy_dtype(dtypes[0]),
    )


# tanhshrink
@handle_frontend_test(
    fn_tree="torch.nn.functional.tanhshrink",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_tanhshrink(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# leaky_relu_
# ToDo test for value test once inplace testing implemented
@handle_frontend_test(
    fn_tree="torch.nn.functional.leaky_relu_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    alpha=st.floats(min_value=0, max_value=1, exclude_min=True),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_leaky_relu_(
    *,
    dtype_and_x,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        input=x[0],
        negative_slope=alpha,
    )


# hardswish
@handle_frontend_test(
    fn_tree="torch.nn.functional.hardswish",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_hardswish(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# hardsigmoid
@handle_frontend_test(
    fn_tree="torch.nn.functional.hardsigmoid",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_hardsigmoid(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    _filter_dtypes(input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# hardtanh
@handle_frontend_test(
    fn_tree="torch.nn.functional.hardtanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    max_val=st.floats(min_value=0, max_value=1, exclude_min=True),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_hardtanh(
    *,
    dtype_and_x,
    max_val,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    max_min = max_val, -max_val
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        min_val=max_min[1],
        max_val=max_min[0],
    )


# hardtanh_
@handle_frontend_test(
    fn_tree="torch.nn.functional.hardtanh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    max_val=st.floats(min_value=0, max_value=1, exclude_min=True),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_hardtanh_(
    *,
    dtype_and_x,
    max_val,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    max_min = max_val, -max_val
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        input=x[0],
        min_val=max_min[1],
        max_val=max_min[0],
    )


# normalize
@handle_frontend_test(
    fn_tree="torch.nn.functional.normalize",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    p=helpers.ints(min_value=2, max_value=5),
)
def test_torch_normalize(
    *,
    dtype_x_and_axis,
    p,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, axis = dtype_x_and_axis
    _filter_dtypes(dtype)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
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
    large_abs_safety_factor=50,
    small_abs_safety_factor=50,
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
    min_dim_size=None,
    max_dim_size=None,
    group=False,
):
    results = draw(
        helpers.dtype_values_axis(
            available_dtypes=available_dtypes,
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale=safety_factor_scale,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
            valid_axis=valid_axis,
            allow_neg_axes=allow_neg_axes,
            max_axes_size=max_axes_size,
            force_int_axis=force_int_axis,
            ret_shape=ret_shape,
        )
    )

    dtype, values, axis, shape = results

    if group:
        channel_size = shape[1]
        group_list = [*range(1, max_dim_size)]
        group_list = list(filter(lambda x: (channel_size % x == 0), group_list))
        group_size = draw(st.sampled_from(group_list))
        weight_shape = [shape[1]]
        bias_shape = [shape[1]]
    else:
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
    if group:
        return dtype, values, weight_values, bias_values, group_size
    return dtype, values, axis, weight_values, bias_values, new_std_values


@handle_frontend_test(
    fn_tree="torch.nn.functional.layer_norm",
    dtype_x_and_axis=_generate_data_layer_norm(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    epsilon=st.floats(min_value=0.01, max_value=0.1),
)
def test_torch_layer_norm(
    *,
    dtype_x_and_axis,
    epsilon,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, axis, weight, bias, new_std = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        input=x[0],
        normalized_shape=axis,
        weight=weight[0],
        bias=bias[0],
        eps=epsilon,
    )


# softplus
@handle_frontend_test(
    fn_tree="torch.nn.functional.softplus",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    beta=st.integers(min_value=1, max_value=20),
    threshold=st.integers(min_value=0, max_value=40),
    test_with_out=st.just(False),
)
def test_torch_softplus(
    *,
    dtype_and_x,
    beta,
    threshold,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        beta=beta,
        threshold=threshold,
    )


# group_norm
@handle_frontend_test(
    fn_tree="torch.nn.functional.group_norm",
    dtype_x_and_axis=_generate_data_layer_norm(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=4,
        group=True,
    ),
    epsilon=st.floats(min_value=0.01, max_value=0.1),
    test_with_out=st.just(False),
)
def test_torch_group_norm(
    dtype_x_and_axis,
    epsilon,
    frontend,
    test_flags,
    fn_tree,
):
    dtype, x, weight, bias, group_size = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=x[0],
        num_groups=group_size,
        weight=weight[0],
        bias=bias[0],
        eps=epsilon,
    )


@st.composite
def _generate_batch_norm_data(draw):
    input_dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=2,
            max_num_dims=4,
            min_dim_size=5,
            ret_shape=True,
            large_abs_safety_factor=25,
            small_abs_safety_factor=25,
            safety_factor_scale="log",
        )
    )
    weight = draw(helpers.array_values(dtype=input_dtype[0], shape=shape[1]))
    bias = draw(helpers.array_values(dtype=input_dtype[0], shape=shape[1]))
    running_mean = draw(helpers.array_values(dtype=input_dtype[0], shape=shape[1]))
    running_var = draw(helpers.array_values(dtype=input_dtype[0], shape=shape[1]))
    return input_dtype, input, weight, bias, running_mean, running_var


# batch_norm
@handle_frontend_test(
    fn_tree="torch.nn.functional.batch_norm",
    data=_generate_batch_norm_data(),
    momentum=helpers.floats(min_value=0.01, max_value=0.1),
    eps=helpers.floats(min_value=1e-5, max_value=0.1),
    training=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_batch_norm(
    *,
    data,
    momentum,
    eps,
    training,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, input, weight, bias, running_mean, running_var = data
    helpers.test_frontend_function(
        input_dtypes=input_dtype * 5,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=input[0],
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


# adaptive_avg_pool1d
@handle_frontend_test(
    fn_tree="torch.nn.functional.adaptive_avg_pool1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=5,
        max_value=100,
        min_value=-100,
    ),
    output_size=helpers.ints(min_value=1, max_value=10),
    test_with_out=st.just(False),
)
def test_torch_adaptive_avg_pool1d(
    *,
    dtype_and_x,
    output_size,
    on_device,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        output_size=output_size,
        atol=1e-2,
    )


# adaptive_avg_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.adaptive_avg_pool2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=4,
        min_dim_size=5,
        max_value=100,
        min_value=-100,
    ),
    output_size=st.tuples(
        helpers.ints(min_value=1, max_value=10),
        helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_torch_adaptive_avg_pool2d(
    *,
    dtype_and_x,
    output_size,
    on_device,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        output_size=output_size,
        atol=1e-2,
    )
