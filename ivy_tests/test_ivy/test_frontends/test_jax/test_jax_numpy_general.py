# global
from hypothesis import strategies as st, assume
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# all
@handle_frontend_test(
    fn_tree="jax.numpy.all",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
    ),
)
def test_jax_numpy_all(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        all_aliases=["numpy.alltrue"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# argmax
@handle_frontend_test(
    fn_tree="jax.numpy.argmax",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        valid_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_jax_numpy_argmax(
    *,
    dtype_and_x,
    keepdims,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
    )


# argwhere
@handle_frontend_test(
    fn_tree="jax.numpy.argwhere",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_numpy_argwhere(
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
        size=None,
        fill_value=None,
    )


# argsort
@handle_frontend_test(
    fn_tree="jax.numpy.argsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_jax_numpy_argsort(
    *,
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
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
    )


# fmax
@handle_frontend_test(
    fn_tree="jax.numpy.fmax",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_jax_numpy_fmax(
    *,
    dtype_and_inputs,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=inputs[0],
        x2=inputs[1],
    )


# bitwise_and
# TODO: add testing for other dtypes
@handle_frontend_test(
    fn_tree="jax.numpy.bitwise_and",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"), num_arrays=2
    ),
)
def test_jax_numpy_bitwise_and(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# bitwise_not
@handle_frontend_test(
    fn_tree="jax.numpy.bitwise_not",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("bool")),
)
def test_jax_numpy_bitwise_not(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# bitwise_or
# TODO: add testing for other dtypes
@handle_frontend_test(
    fn_tree="jax.numpy.bitwise_or",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"), num_arrays=2
    ),
)
def test_jax_numpy_bitwise_or(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# bitwise_xor
# TODO: add testing for other dtypes
@handle_frontend_test(
    fn_tree="jax.numpy.bitwise_xor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"), num_arrays=2
    ),
)
def test_jax_numpy_bitwise_xor(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# heaviside
@handle_frontend_test(
    fn_tree="jax.numpy.heaviside",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_heaviside(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


# any
@handle_frontend_test(
    fn_tree="jax.numpy.any",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        valid_axis=True,
        max_axes_size=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
    where=np_helpers.where(),
)
def test_jax_numpy_any(
    *,
    dtype_x_axis,
    keepdims,
    where,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, as_variable, native_array = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        all_aliases=["numpy.sometrue"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
    )


# hstack
@handle_frontend_test(
    fn_tree="jax.numpy.hstack",
    dtype_and_tup=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shared_dtype=True,
        num_arrays=st.integers(min_value=2, max_value=2),
        shape=helpers.get_shape(
            min_num_dims=1, max_num_dims=3, min_dim_size=1, max_dim_size=5
        ),
    ),
)
def test_jax_numpy_hstack(
    dtype_and_tup,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
):
    input_dtype, x = dtype_and_tup
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        tup=x,
    )


# maximum
@handle_frontend_test(
    fn_tree="jax.numpy.maximum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_maximum(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# minimum
@handle_frontend_test(
    fn_tree="jax.numpy.minimum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.minimum"
    ),
)
def test_jax_numpy_minimum(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# diagonal
@st.composite
def dims_and_offset(draw, shape):
    shape_actual = draw(shape)
    dim1 = draw(helpers.get_axis(shape=shape, force_int=True))
    dim2 = draw(helpers.get_axis(shape=shape, force_int=True))
    offset = draw(
        st.integers(min_value=-shape_actual[dim1], max_value=shape_actual[dim1])
    )
    return dim1, dim2, offset


@handle_frontend_test(
    fn_tree="jax.numpy.diagonal",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dims_and_offset=dims_and_offset(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape")
    ),
)
def test_jax_numpy_diagonal(
    *,
    dtype_and_values,
    dims_and_offset,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    axis1, axis2, offset = dims_and_offset
    a = value[0]
    num_of_dims = len(np.shape(a))
    assume(axis1 != axis2)
    if axis1 < 0:
        assume(axis1 + num_of_dims != axis2)
    if axis2 < 0:
        assume(axis1 != axis2 + num_of_dims)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


# eye
@handle_frontend_test(
    fn_tree="jax.numpy.eye",
    n=helpers.ints(min_value=3, max_value=10),
    m=st.none() | helpers.ints(min_value=3, max_value=10),
    k=helpers.ints(min_value=-2, max_value=2),
    dtypes=helpers.get_dtypes("valid", full=False),
)
def test_jax_numpy_eye(
    *,
    n,
    m,
    k,
    dtypes,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        N=n,
        M=m,
        k=k,
        dtype=dtypes[0],
    )


# diag
@st.composite
def _diag_helper(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            small_abs_safety_factor=2,
            large_abs_safety_factor=2,
            safety_factor_scale="log",
            min_num_dims=1,
            max_num_dims=2,
            min_dim_size=1,
            max_dim_size=50,
        )
    )
    shape = x[0].shape
    if len(shape) == 2:
        k = draw(helpers.ints(min_value=-shape[0] + 1, max_value=shape[1] - 1))
    else:
        k = draw(helpers.ints(min_value=0, max_value=shape[0]))
    return dtype, x, k


@handle_frontend_test(
    fn_tree="jax.numpy.diag",
    dtype_x_k=_diag_helper(),
)
def test_jax_numpy_diag(
    *,
    dtype_x_k,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x, k = dtype_x_k
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        v=x[0],
        k=k,
    )


# zeros_like
@handle_frontend_test(
    fn_tree="jax.numpy.zeros_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_numpy_zeros_like(
    dtype_and_x,
    dtype,
    shape,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
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
        dtype=dtype[0],
        shape=shape,
    )


# msort
@handle_frontend_test(
    fn_tree="jax.numpy.msort",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
)
def test_jax_numpy_msort(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        a=x[0],
    )


# arange
@handle_frontend_test(
    fn_tree="jax.numpy.arange",
    start=st.integers(min_value=-100, max_value=100),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_jax_numpy_arange(
    *,
    start,
    stop,
    step,
    dtype,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
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
        start=start,
        stop=stop,
        step=step,
        dtype=dtype[0],
    )


# ones
@handle_frontend_test(
    fn_tree="jax.numpy.ones",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_jax_numpy_ones(
    shape,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
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


# ones_like
@handle_frontend_test(
    fn_tree="jax.numpy.ones_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_jax_numpy_ones_like(
    dtype_and_x,
    shape,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
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
        dtype=dtype[0],
        shape=shape,
    )
