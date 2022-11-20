# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _get_clip_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    min = draw(
        helpers.array_values(dtype=x_dtype[0], shape=shape, min_value=-50, max_value=5)
    )
    max = draw(
        helpers.array_values(dtype=x_dtype[0], shape=shape, min_value=6, max_value=50)
    )
    return x_dtype, x, min, max


# clip_by_value
@handle_frontend_test(
    fn_tree="tensorflow.clip_by_value",
    input_and_ranges=_get_clip_inputs(),
)
def test_tensorflow_clip_by_value(
    *,
    input_and_ranges,
    as_variable,
    native_array,
    frontend,
    fn_tree,
    on_device,
    num_positional_args,
):
    x_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        t=x[0],
        clip_value_min=min,
        clip_value_max=max,
    )


# eye
@handle_frontend_test(
    fn_tree="tensorflow.eye",
    n_rows=helpers.ints(min_value=0, max_value=10),
    n_cols=st.none() | helpers.ints(min_value=0, max_value=10),
    batch_shape=st.lists(
        helpers.ints(min_value=1, max_value=10), min_size=1, max_size=2
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_tensorflow_eye(
    *,
    n_rows,
    n_cols,
    batch_shape,
    dtype,
    as_variable,
    native_array,
    with_out,
    frontend,
    fn_tree,
    on_device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        num_rows=n_rows,
        num_columns=n_cols,
        batch_shape=batch_shape,
        dtype=dtype[0],
    )


# ones
@handle_frontend_test(
    fn_tree="tensorflow.ones",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_tensorflow_ones(
    shape,
    dtype,
    as_variable,
    native_array,
    num_positional_args,
    frontend,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype[0],
    )
