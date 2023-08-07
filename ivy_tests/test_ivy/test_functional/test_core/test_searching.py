"""Collection of tests for searching functions."""

# Global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# Helpers #
############


@st.composite
def _dtype_x_limited_axis(draw, *, allow_none=False):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=1,
            min_dim_size=1,
            ret_shape=True,
        )
    )
    if allow_none and draw(st.booleans()):
        return dtype, x, None

    axis = draw(helpers.ints(min_value=0, max_value=len(shape) - 1))
    return dtype, x, axis


@st.composite
def _broadcastable_trio(draw):
    shape = draw(helpers.get_shape(min_num_dims=1, min_dim_size=1))
    cond = draw(helpers.array_values(dtype="bool", shape=shape))
    dtypes, xs = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            num_arrays=2,
            shape=shape,
            shared_dtype=True,
            large_abs_safety_factor=16,
            small_abs_safety_factor=16,
            safety_factor_scale="log",
        )
    )
    return cond, xs, dtypes


# Functions #
#############


@handle_test(
    fn_tree="functional.ivy.argmax",
    dtype_x_axis=_dtype_x_limited_axis(allow_none=True),
    keepdims=st.booleans(),
    dtype=helpers.get_dtypes("integer", full=False, none=True),
    select_last_index=st.booleans(),
)
def test_argmax(
    *,
    dtype_x_axis,
    keepdims,
    dtype,
    select_last_index,
    test_flags,
    backend_fw,
    fn_name,
    on_device
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        axis=axis,
        keepdims=keepdims,
        dtype=dtype[0],
        select_last_index=select_last_index,
    )


@handle_test(
    fn_tree="functional.ivy.argmin",
    dtype_x_axis=_dtype_x_limited_axis(allow_none=True),
    keepdims=st.booleans(),
    output_dtype=helpers.get_dtypes("integer", full=False, none=True),
    select_last_index=st.booleans(),
)
def test_argmin(
    *,
    dtype_x_axis,
    keepdims,
    output_dtype,
    select_last_index,
    test_flags,
    backend_fw,
    fn_name,
    on_device
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        axis=axis,
        keepdims=keepdims,
        dtype=output_dtype[0],
        select_last_index=select_last_index,
    )


@handle_test(
    fn_tree="functional.ivy.nonzero",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    as_tuple=st.booleans(),
    size=st.integers(min_value=1, max_value=5),
    fill_value=st.one_of(st.integers(0, 5), helpers.floats()),
    test_with_out=st.just(False),
)
def test_nonzero(
    *,
    dtype_and_x,
    as_tuple,
    size,
    fill_value,
    test_flags,
    backend_fw,
    fn_name,
    on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        as_tuple=as_tuple,
        size=size,
        fill_value=fill_value,
    )


@handle_test(
    fn_tree="functional.ivy.where",
    broadcastables=_broadcastable_trio(),
)
def test_where(*, broadcastables, test_flags, backend_fw, fn_name, on_device):
    cond, xs, dtypes = broadcastables

    helpers.test_function(
        input_dtypes=["bool"] + dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        condition=cond,
        x1=xs[0],
        x2=xs[1],
    )


# argwhere
@handle_test(
    fn_tree="functional.ivy.argwhere",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=("bool",)),
    ground_truth_backend="torch",
)
def test_argwhere(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )
