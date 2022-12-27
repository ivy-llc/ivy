import hypothesis.extra.numpy as hnp
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _broadcastable_trio(draw):
    dtype = draw(helpers.get_dtypes("valid", full=False))
    shapes_st = draw(
        hnp.mutually_broadcastable_shapes(num_shapes=3, min_dims=1, min_side=1)
    )
    cond_shape, x1_shape, x2_shape = shapes_st.input_shapes
    cond = draw(hnp.arrays(hnp.boolean_dtypes(), cond_shape))
    x1 = draw(helpers.array_values(dtype=dtype[0], shape=x1_shape))
    x2 = draw(helpers.array_values(dtype=dtype[0], shape=x2_shape))
    return cond, x1, x2, (dtype * 2)


# where
@handle_frontend_test(
    fn_tree="numpy.where",
    broadcastables=_broadcastable_trio(),
)
def test_numpy_where(
    broadcastables,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    cond, x1, x2, dtype = broadcastables
    helpers.test_frontend_function(
        input_dtypes=["bool", dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        cond=cond,
        x1=x1,
        x2=x2,
    )


# nonzero
@handle_frontend_test(
    fn_tree="numpy.nonzero",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_nonzero(
    dtype_and_a,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
    )


# argmin
@handle_frontend_test(
    fn_tree="numpy.argmin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_numpy_argmin(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
        out=None,
    )


# argmax
@handle_frontend_test(
    fn_tree="numpy.argmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_numpy_argmax(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
        out=None,
    )


# flatnonzero
@handle_frontend_test(
    fn_tree="numpy.flatnonzero",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_numpy_flatnonzero(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# searchsorted
@handle_frontend_test(
    fn_tree="numpy.searchsorted",
    dtype_x_v=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
    ),
    side=st.sampled_from(["left", "right"]),
)
def test_numpy_searchsorted(
    dtype_x_v,
    side,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtypes, xs = dtype_x_v
    helpers.test_frontend_function(
        input_dtypes=input_dtypes + ["int64"],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        v=xs[1],
        side=side,
        sorter=np.argsort(xs[0]),
    )


# argwhere
@handle_frontend_test(
    fn_tree="numpy.argwhere",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_argwhere(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# nanargmax
@handle_frontend_test(
    fn_tree="numpy.nanargmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_numpy_nanargmax(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


# nanargmin
@handle_frontend_test(
    fn_tree="numpy.nanargmin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    keep_dims=st.booleans(),
)
def test_numpy_nanargmin(
    dtype_x_axis,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
    )
