# global
import ivy
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Helper functions
@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(draw(helpers.get_dtypes("numeric"))), length=1
            ),
            key="dtype",
        )
    )


@st.composite
def _fill_value(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_uint_dtype(dtype):
        return draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype):
        return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


@st.composite
def _requires_grad(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype):
        return draw(st.just(False))
    return draw(st.booleans())


# full
@handle_frontend_test(
    fn_tree="torch.full",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
)
def test_torch_full(
    *,
    shape,
    fill_value,
    dtypes,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        size=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        requires_grad=requires_grad,
        device=on_device,
    )


# ones_like
@handle_frontend_test(
    fn_tree="torch.ones_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
)
def test_torch_ones_like(
    *,
    dtype_and_x,
    dtypes,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dtype=dtypes[0],
        requires_grad=requires_grad,
        device=on_device,
    )


# ones
@handle_frontend_test(
    fn_tree="torch.ones",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
)
def test_torch_ones(
    *,
    shape,
    dtypes,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        size=shape,
        dtype=dtypes[0],
        requires_grad=requires_grad,
        device=on_device,
    )


# zeros
@handle_frontend_test(
    fn_tree="torch.zeros",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
)
def test_torch_zeros(
    *,
    shape,
    dtypes,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        size=shape,
        dtype=dtypes[0],
        requires_grad=requires_grad,
        device=on_device,
    )


# zeros_like
@handle_frontend_test(
    fn_tree="torch.zeros_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
)
def test_torch_zeros_like(
    *,
    dtype_and_x,
    dtypes,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dtype=dtypes[0],
        requires_grad=requires_grad,
        device=on_device,
    )


# empty
@handle_frontend_test(
    fn_tree="torch.empty",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=helpers.get_dtypes("valid", full=False),
    requires_grad=_requires_grad(),
)
def test_torch_empty(
    *,
    shape,
    dtypes,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        size=shape,
        dtype=dtypes,
        requires_grad=requires_grad,
        device=on_device,
    )


# arange
@handle_frontend_test(
    fn_tree="torch.arange",
    start=helpers.ints(min_value=0, max_value=50),
    stop=helpers.ints(min_value=0, max_value=50),
    step=helpers.ints(min_value=-50, max_value=50).filter(
        lambda x: True if x != 0 else False
    ),
    dtypes=helpers.get_dtypes("float", full=False),
    requires_grad=_requires_grad(),
)
def test_torch_arange(
    *,
    start,
    stop,
    step,
    dtypes,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        end=stop,
        start=start,
        step=step,
        dtype=dtypes[0],
        requires_grad=requires_grad,
        device=on_device,
    )


# range
@handle_frontend_test(
    fn_tree="torch.range",
    start=helpers.ints(min_value=0, max_value=50),
    stop=helpers.ints(min_value=0, max_value=50),
    step=helpers.ints(min_value=-50, max_value=50).filter(
        lambda x: True if x != 0 else False
    ),
    dtypes=helpers.get_dtypes("float", full=False),
    requires_grad=_requires_grad(),
)
def test_torch_range(
    *,
    start,
    stop,
    step,
    dtypes,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        end=stop,
        start=start,
        step=step,
        dtype=dtypes,
        requires_grad=requires_grad,
        device=on_device,
    )


# linspace
@handle_frontend_test(
    fn_tree="torch.linspace",
    start=st.floats(min_value=-10, max_value=10),
    stop=st.floats(min_value=-10, max_value=10),
    num=st.integers(min_value=1, max_value=10),
    requires_grad=_requires_grad(),
)
def test_torch_linspace(
    *,
    start,
    stop,
    num,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        steps=num,
        requires_grad=requires_grad,
        device=on_device,
    )


# logspace
@handle_frontend_test(
    fn_tree="torch.logspace",
    start=st.floats(min_value=-10, max_value=10),
    stop=st.floats(min_value=-10, max_value=10),
    num=st.integers(min_value=1, max_value=10),
    requires_grad=_requires_grad(),
)
def test_torch_logspace(
    *,
    start,
    stop,
    num,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        steps=num,
        requires_grad=requires_grad,
        device=on_device,
    )


# empty_like
@handle_frontend_test(
    fn_tree="torch.empty_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    requires_grad=_requires_grad(),
)
def test_torch_empty_like(
    *,
    dtype_and_x,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, inputs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        input=inputs[0],
        dtype=dtype[0],
        requires_grad=requires_grad,
        device=on_device,
        test_values=False,
    )


# full_like
@handle_frontend_test(
    fn_tree="torch.full_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
)
def test_torch_full_like(
    *,
    dtype_and_x,
    fill_value,
    dtypes,
    requires_grad,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, inputs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        input=inputs[0],
        fill_value=fill_value,
        dtype=dtypes[0],
        requires_grad=requires_grad,
        on_device=on_device,
        test_values=False,
    )


# as_tensor and tensor by proxy
@handle_frontend_test(
    fn_tree="torch.as_tensor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_torch_as_tensor(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        data=input[0],
        device=on_device,
    )


# from_numpy
@handle_frontend_test(
    fn_tree="torch.from_numpy",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_torch_from_numpy(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        data=input[0],
    )
