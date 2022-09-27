# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# abs
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.abs"
    ),
)
def test_jax_numpy_abs(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.abs",
        x=x[0],
    )


# absolute
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.absolute"
    ),
)
def test_jax_numpy_absolute(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.absolute",
        x=x[0],
    )


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.add"
    ),
)
def test_jax_numpy_add(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.add",
        x1=x[0],
        x2=x[0],
    )


# all
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.all"
    ),
)
def test_jax_numpy_all(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.all",
        a=x[0],
    )


# allclose
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.allclose"
    ),
    equal_nan=st.booleans(),
)
def test_jax_numpy_allclose(
    dtype_and_input,
    equal_nan,
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
        frontend="jax",
        fn_tree="numpy.allclose",
        a=input[0],
        b=input[1],
        rtol=1e-05,
        atol=1e-08,
        equal_nan=equal_nan,
    )


# broadcast_to
@st.composite
def _get_input_and_broadcast_shape(draw):
    dim1 = draw(helpers.ints(min_value=2, max_value=5))
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=10,
            shape=(dim1,),
        )
    )
    broadcast_dim = draw(helpers.ints(min_value=1, max_value=3))
    shape = ()
    for _ in range(broadcast_dim):
        shape += (draw(helpers.ints(min_value=1, max_value=dim1)),)
    shape += (dim1,)
    return x_dtype, x, shape


@handle_cmd_line_args
@given(
    input_x_broadcast=_get_input_and_broadcast_shape(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.broadcast_to"
    ),
)
def test_jax_numpy_broadcast_to(
    input_x_broadcast,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    x_dtype, x, shape = input_x_broadcast
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        with_out=False,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.broadcast_to",
        arr=x[0],
        shape=shape,
    )


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
    min = draw(st.booleans())
    if min:
        max = draw(st.booleans())
        min = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=-50, max_value=5
            )
        )
        max = (
            draw(
                helpers.array_values(
                    dtype=x_dtype[0], shape=shape, min_value=6, max_value=50
                )
            )
            if max
            else None
        )
    else:
        min = None
        max = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=6, max_value=50
            )
        )
    return x_dtype, x, min, max


# clip
@handle_cmd_line_args
@given(
    input_and_ranges=_get_clip_inputs(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.clip"
    ),
)
def test_jax_numpy_clip(
    input_and_ranges,
    num_positional_args,
    with_out,
    as_variable,
    native_array,
    fw,
):
    x_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        with_out=with_out,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.clip",
        a=x[0],
        a_min=min,
        a_max=max,
        out=None,
    )


# reshape
@st.composite
def _get_input_and_reshape(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=10,
            shape=shape,
        )
    )
    new_shape = shape[1:] + (shape[0],)
    return x_dtype, x, new_shape


@handle_cmd_line_args
@given(
    input_x_shape=_get_input_and_reshape(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.reshape"
    ),
)
def test_jax_numpy_reshape(
    input_x_shape,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    x_dtype, x, shape = input_x_shape
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        with_out=False,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.reshape",
        a=x[0],
        newshape=shape,
    )


@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )
    xs = list()
    input_dtypes = draw(
        helpers.array_dtypes(available_dtypes=draw(helpers.get_dtypes("valid")))
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


@st.composite
def _dtype_n_with_out(draw):
    dtype = draw(helpers.get_dtypes("float", none=True))
    if dtype is None:
        return dtype, draw(st.booleans())
    return dtype, False


@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    dtype_n_with_out=_dtype_n_with_out(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.concatenate"
    ),
)
def test_jax_numpy_concat(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    dtype_n_with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, with_out = dtype_n_with_out
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.concatenate",
        arrays=xs,
        axis=unique_idx,
        out=None,
        dtype=dtype,
    )
