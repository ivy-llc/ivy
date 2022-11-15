# local
import ivy_tests.test_ivy.helpers as helpers
from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers import handle_frontend_test


# squeeze
@st.composite
def _squeeze_helper(draw):
    shape = draw(st.shared(helpers.get_shape(), key="value_shape"))
    valid_axes = []
    for index, axis in enumerate(shape):
        if axis == 1:
            valid_axes.append(index)
    valid_axes.insert(0, None)

    return draw(st.sampled_from(valid_axes))


@handle_frontend_test(
    fn_tree="numpy.squeeze",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=_squeeze_helper(),
)
def test_numpy_squeeze(
    dtype_and_x,
    axis,
    num_positional_args,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        frontend="numpy",
        fn_tree=fn_tree,
        a=x[0],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="numpy.expand_dims",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="value_shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
)
def test_numpy_expand_dims(
    dtype_and_x,
    axis,
    num_positional_args,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        frontend="numpy",
        fn_tree=fn_tree,
        a=x[0],
        axis=axis,
    )
