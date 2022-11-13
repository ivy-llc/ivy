# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


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
@handle_cmd_line_args
@given(
    input_and_ranges=_get_clip_inputs(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.clip_by_value"
    ),
)
def test_tensorflow_clip_by_value(
    input_and_ranges,
    num_positional_args,
    as_variable,
    native_array,
):
    x_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="clip_by_value",
        t=x[0],
        clip_value_min=min,
        clip_value_max=max,
    )


# eye
@handle_cmd_line_args
@given(
    n_rows=helpers.ints(min_value=0, max_value=10),
    n_cols=st.none() | helpers.ints(min_value=0, max_value=10),
    batch_shape=st.lists(
        helpers.ints(min_value=1, max_value=10), min_size=1, max_size=2
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.eye"
    ),
)
def test_tensorflow_eye(
    n_rows,
    n_cols,
    batch_shape,
    dtype,
    as_variable,
    native_array,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="eye",
        num_rows=n_rows,
        num_columns=n_cols,
        batch_shape=batch_shape,
        dtype=dtype,
    )


# sort
@handle_cmd_line_args
@given(
    values=helpers.array_values(
        dtype=helpers.get_dtypes("valid"), 
        shape=(5, 6), 
        min_value=1, 
        max_value=9
        ),
    axis=helpers.ints(),
    direction=helpers.subsets(elements=["ASCENDING", "DESCENDING"]),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.sort"
    ),
    datatype=helpers.get_dtypes("valid")
)
def test_tensorflow_sort(
    values,
    axis,
    direction,
    as_variable,
    native_array,
    num_positional_args,
    datatype
):
    helpers.test_frontend_function(
        input_dtypes=datatype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="sort",
        values=values,
        direction=direction,
        axis=axis,
    )

