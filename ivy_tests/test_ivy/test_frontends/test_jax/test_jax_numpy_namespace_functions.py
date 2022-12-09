# global
from hypothesis import strategies as st, assume
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
    _get_castable_dtype,
)
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_value1_value2_axis_for_tensordot,
)


# absolute
@handle_frontend_test(
    fn_tree="jax.numpy.absolute",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
)
def test_jax_numpy_absolute(
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
        all_aliases=["jax.numpy.abs"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
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


# add
@handle_frontend_test(
    fn_tree="jax.numpy.add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2, shared_dtype=True
    ),
)
def test_jax_numpy_add(
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
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


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


# tan
@handle_frontend_test(
    fn_tree="jax.numpy.tan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_tan(
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


# allclose
@handle_frontend_test(
    fn_tree="jax.numpy.allclose",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    rtol=st.floats(min_value=1e-5, max_value=1e-1),
    atol=st.floats(min_value=1e-8, max_value=1e-6),
    equal_nan=st.booleans(),
)
def test_jax_numpy_allclose(
        *,
        dtype_and_input,
        equal_nan,
        rtol,
        atol,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=rtol,
        atol=atol,
        a=input[0],
        b=input[1],
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


@handle_frontend_test(
    fn_tree="jax.numpy.broadcast_to",
    input_x_broadcast=_get_input_and_broadcast_shape(),
)
def test_jax_numpy_broadcast_to(
        *,
        input_x_broadcast,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    x_dtype, x, shape = input_x_broadcast
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
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
            min_value=-1e10,
            max_value=1e10,
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
@handle_frontend_test(
    fn_tree="jax.numpy.clip",
    input_and_ranges=_get_clip_inputs(),
)
def test_jax_numpy_clip(
        *,
        input_and_ranges,
        num_positional_args,
        as_variable,
        native_array,
        with_out,
        on_device,
        fn_tree,
        frontend,
):
    x_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        a_min=min,
        a_max=max,
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


@handle_frontend_test(
    fn_tree="jax.numpy.reshape",
    input_x_shape=_get_input_and_reshape(),
    order=st.sampled_from(["C", "F"]),
)
def test_jax_numpy_reshape(
        *,
        input_x_shape,
        order,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    x_dtype, x, shape = input_x_shape
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        newshape=shape,
        order=order,
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


@handle_frontend_test(
    fn_tree="jax.numpy.concatenate",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
)
def test_jax_numpy_concat(
        *,
        xs_n_input_dtypes_n_unique_idx,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        arrays=xs,
        axis=unique_idx,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.mean",
    dtype_x_axis=statistical_dtype_values(function="mean"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_mean(
        *,
        dtype_x_axis,
        dtype,
        keepdims,
        where,
        num_positional_args,
        with_out,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    x_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, as_variable, native_array = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=x_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )

    np_helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-2,
        rtol=1e-2,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        keepdims=keepdims,
        where=where,
    )


# uint16
@handle_frontend_test(
    fn_tree="jax.numpy.uint16",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_jax_numpy_uint16(
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
    if ivy.current_backend_str() != "torch":
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


# var
@handle_frontend_test(
    fn_tree="jax.numpy.var",
    dtype_x_axis=statistical_dtype_values(function="var"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_var(
        *,
        dtype_x_axis,
        dtype,
        keepdims,
        where,
        num_positional_args,
        with_out,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    x_dtype, x, axis, ddof = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, as_variable, native_array = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=x_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )

    np_helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
    )


# dot
@st.composite
def _get_dtype_input_and_vectors(draw):
    dim_size = draw(helpers.ints(min_value=1, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    if dim_size == 1:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
    else:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
    return dtype, vec1, vec2


# mod
@handle_frontend_test(
    fn_tree="jax.numpy.mod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_mod(
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
    assume(not np.any(np.isclose(x[1], 0)))
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


@handle_frontend_test(
    fn_tree="jax.numpy.dot",
    dtype_x_y=_get_dtype_input_and_vectors(),
)
def test_jax_numpy_dot(
        *,
        dtype_x_y,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, x, y = dtype_x_y
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        rtol=1e-01,
        atol=1e-01,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x,
        b=y,
        precision=None,
    )


# einsum
@handle_frontend_test(
    fn_tree="jax.numpy.einsum",
    eq_n_op=st.sampled_from(
        [
            (
                "ii",
                np.arange(25).reshape(5, 5),
            ),
            (
                "ii->i",
                np.arange(25).reshape(5, 5),
            ),
            ("ij,j", np.arange(25).reshape(5, 5), np.arange(5)),
        ]
    ),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_jax_numpy_einsum(
        *,
        eq_n_op,
        dtype,
        as_variable,
        native_array,
        with_out,
        on_device,
        fn_tree,
        frontend,
):
    kw = {}
    i = 0
    for x_ in eq_n_op:
        kw["x{}".format(i)] = x_
        i += 1
    num_positional_args = i
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        **kw,
        out=None,
        optimize="optimal",
        precision=None,
        _use_xeinsum=False,
    )


# arctan
@handle_frontend_test(
    fn_tree="jax.numpy.arctan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_arctan(
        *,
        dtype_and_x,
        num_positional_args,
        as_variable,
        with_out,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arctan2
@handle_frontend_test(
    fn_tree="jax.numpy.arctan2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_arctan2(
        *,
        dtype_and_x,
        num_positional_args,
        as_variable,
        with_out,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.cos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_cos(
        *,
        dtype_and_x,
        num_positional_args,
        as_variable,
        with_out,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# cosh
@handle_frontend_test(
    fn_tree="jax.numpy.cosh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_cosh(
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


# tanh
@handle_frontend_test(
    fn_tree="jax.numpy.tanh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_tanh(
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


# sinh
@handle_frontend_test(
    fn_tree="jax.numpy.sinh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_sinh(
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


@handle_frontend_test(
    fn_tree="jax.numpy.sin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_sin(
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


# floor
@handle_frontend_test(
    fn_tree="jax.numpy.floor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_floor(
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


@handle_frontend_test(
    fn_tree="jax.numpy.array_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=-np.inf,
        max_value=np.inf,
        shared_dtype=True,
    ),
    equal_nan=st.booleans(),
)
def test_jax_numpy_array_equal(
        *,
        dtype_and_x,
        equal_nan,
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
        a1=x[0],
        a2=x[1],
        equal_nan=equal_nan,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.array_equiv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_array_equiv(
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
        a1=x[0],
        a2=x[1],
    )


# zeros
@handle_frontend_test(
    fn_tree="jax.numpy.zeros",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=helpers.get_dtypes("numeric", full=False),
)
def test_jax_numpy_zeros(
        *,
        dtypes,
        shape,
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
        shape=shape,
        dtype=dtypes[0],
    )


# arccos
@handle_frontend_test(
    fn_tree="jax.numpy.arccos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_arccos(
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


# arccosh
@handle_frontend_test(
    fn_tree="jax.numpy.arccosh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_arccosh(
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


# arcsin
@handle_frontend_test(
    fn_tree="jax.numpy.arcsin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_arcsin(
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


# arcsinh
@handle_frontend_test(
    fn_tree="jax.numpy.arcsinh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_arcsinh(
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


# argmin
@handle_frontend_test(
    fn_tree="jax.numpy.argmin",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        valid_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_jax_numpy_argmin(
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


# moveaxis
@handle_frontend_test(
    fn_tree="jax.numpy.moveaxis",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-100,
        max_value=100,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
    ),
    source=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
    destination=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
)
def test_jax_numpy_moveaxis(
        *,
        dtype_and_a,
        source,
        destination,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
        source=source,
        destination=destination,
    )


# flipud
@handle_frontend_test(
    fn_tree="jax.numpy.flipud",
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_jax_numpy_flipud(
        *,
        dtype_and_m,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, m = dtype_and_m
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        m=m[0],
    )


# power
@handle_frontend_test(
    fn_tree="jax.numpy.power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
)
def test_jax_numpy_power(
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


# bincount
@handle_frontend_test(
    fn_tree="jax.numpy.bincount",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1,
        max_value=2,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=1,
            ),
            key="a_s_d",
        ),
    ),
)
def test_jax_numpy_bincount(
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
        weights=None,
        minlength=0,
        length=None,
    )


# cumprod
@handle_frontend_test(
    fn_tree="jax.numpy.cumprod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_value=-100,
        max_value=100,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", none=True, full=False),
)
def test_jax_numpy_cumprod(
        *,
        dtype_x_axis,
        dtype,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        all_aliases=["numpy.cumproduct"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
    )


# trunc
@handle_frontend_test(
    fn_tree="jax.numpy.trunc",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_trunc(
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


@handle_frontend_test(
    fn_tree="jax.numpy.ceil",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_ceil(
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


# float_power
@handle_frontend_test(
    fn_tree="jax.numpy.float_power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_jax_numpy_float_power(
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


# cumsum
@handle_frontend_test(
    fn_tree="jax.numpy.cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_value=-100,
        max_value=100,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("numeric", none=True, full=False),
)
def test_jax_numpy_cumsum(
        *,
        dtype_x_axis,
        dtype,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
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
        dtype=dtype[0],
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


# deg2rad
@handle_frontend_test(
    fn_tree="jax.numpy.deg2rad",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_numpy_deg2rad(
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


# exp2
@handle_frontend_test(
    fn_tree="jax.numpy.exp2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_jax_numpy_exp2(
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
        rtol=1e-01,
        atol=1e-02,
    )


# gcd
@handle_frontend_test(
    fn_tree="jax.numpy.gcd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
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
def test_jax_numpy_gcd(
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


# i0
@handle_frontend_test(
    fn_tree="jax.numpy.i0",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_jax_numpy_i0(
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


# isneginf
@handle_frontend_test(
    fn_tree="jax.numpy.isneginf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
)
def test_jax_numpy_isneginf(
        *,
        dtype_and_x,
        with_out,
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
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# isposinf
@handle_frontend_test(
    fn_tree="jax.numpy.isposinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
)
def test_jax_numpy_isposinf(
        *,
        dtype_and_x,
        with_out,
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
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# kron
@handle_frontend_test(
    fn_tree="jax.numpy.kron",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_kron(
        *,
        dtype_x,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, x = dtype_x
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
        b=x[1],
    )


# sum
@handle_frontend_test(
    fn_tree="jax.numpy.sum",
    dtype_x_axis_castable=_get_castable_dtype(),
    initial=st.none() | st.floats(-10.0, 10.0),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_sum(
        *,
        dtype_x_axis_castable,
        initial,
        where,
        keepdims,
        with_out,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    x_dtype, x, axis, castable_dtype = dtype_x_axis_castable

    if isinstance(axis, tuple):
        axis = axis[0]
    where, as_variable, native_array = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=x_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )

    np_helpers.test_frontend_function(
        input_dtypes=[x_dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-2,
        a=x[0],
        axis=axis,
        dtype=castable_dtype,
        out=None,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


# lcm
@handle_frontend_test(
    fn_tree="jax.numpy.lcm",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_jax_numpy_lcm(
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


# logaddexp2
@handle_frontend_test(
    fn_tree="jax.numpy.logaddexp2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_jax_numpy_logaddexp2(
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
        rtol=1e-01,
        atol=1e-02,
        x1=x[0],
        x2=x[1],
    )


# trapz
@st.composite
def _either_x_dx(draw):
    dtype_values_axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=st.shared(helpers.get_dtypes("float"), key="trapz_dtype"),
            min_value=-100,
            max_value=100,
            min_num_dims=1,
            max_num_dims=3,
            min_dim_size=1,
            max_dim_size=3,
            allow_neg_axes=True,
            valid_axis=True,
            force_int_axis=True,
        ),
    )
    rand = (draw(st.integers(min_value=0, max_value=1)),)
    if rand == 0:
        either_x_dx = draw(
            helpers.dtype_and_x(
                avaliable_dtypes=st.shared(
                    helpers.get_dtypes("float"), key="trapz_dtype"
                ),
                min_value=-100,
                max_value=100,
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            )
        )
        return dtype_values_axis, rand, either_x_dx
    else:
        either_x_dx = draw(
            st.floats(min_value=-10, max_value=10),
        )
        return dtype_values_axis, rand, either_x_dx


@handle_frontend_test(
    fn_tree="jax.numpy.trapz",
    dtype_x_axis_rand_either=_either_x_dx(),
)
def test_jax_numpy_trapz(
        *,
        dtype_x_axis_rand_either,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    dtype_values_axis, rand, either_x_dx = dtype_x_axis_rand_either
    input_dtype, y, axis = dtype_values_axis
    if rand == 0:
        dtype_x, x = either_x_dx
        x = np.asarray(x, dtype=dtype_x)
        dx = None
    else:
        x = None
        dx = either_x_dx
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        y=y[0],
        x=x,
        dx=dx,
        axis=axis,
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


# flip
@handle_frontend_test(
    fn_tree="jax.numpy.flip",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
)
def test_jax_numpy_flip(
        *,
        dtype_value,
        axis,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        m=value[0],
        axis=axis,
    )


# fliplr
@handle_frontend_test(
    fn_tree="jax.numpy.fliplr",
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
    ),
)
def test_jax_numpy_fliplr(
        *,
        dtype_and_m,
        num_positional_args,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, m = dtype_and_m
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        m=m[0],
    )


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


# arctanh
@handle_frontend_test(
    fn_tree="jax.numpy.arctanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=0,
    ),
)
def test_jax_numpy_arctanh(
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
        x=x[0],
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


# multiply
@handle_frontend_test(
    fn_tree="jax.numpy.multiply",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_multiply(
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


# not_equal
@handle_frontend_test(
    fn_tree="jax.numpy.not_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_not_equal(
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


# less
@handle_frontend_test(
    fn_tree="jax.numpy.less",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_less(
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


# less_equal
@handle_frontend_test(
    fn_tree="jax.numpy.less_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_less_equal(
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


# greater
@handle_frontend_test(
    fn_tree="jax.numpy.greater",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_greater(
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


# greater_equal
@handle_frontend_test(
    fn_tree="jax.numpy.greater_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_greater_equal(
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


# equal
@handle_frontend_test(
    fn_tree="jax.numpy.equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_numpy_equal(
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


# min
@handle_frontend_test(
    fn_tree="jax.numpy.min",
    dtype_x_axis=statistical_dtype_values(function="min"),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_min(
        *,
        dtype_x_axis,
        keepdims,
        where,
        num_positional_args,
        with_out,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    x_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, as_variable, native_array = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=x_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )

    np_helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        all_aliases=["numpy.amin"],
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


# max
@handle_frontend_test(
    fn_tree="jax.numpy.max",
    dtype_x_axis=statistical_dtype_values(function="max"),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_max(
        *,
        dtype_x_axis,
        keepdims,
        where,
        num_positional_args,
        with_out,
        as_variable,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    x_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, as_variable, native_array = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=x_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )

    np_helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        all_aliases=["numpy.amax"],
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


# log10
@handle_frontend_test(
    fn_tree="jax.numpy.log10",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
    ),
)
def test_jax_numpy_log10(
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
        rtol=1e-01,
        atol=1e-02,
        x=x[0],
    )


# logaddexp
@handle_frontend_test(
    fn_tree="jax.numpy.logaddexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_jax_numpy_logaddexp(
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
        rtol=1e-01,
        atol=1e-02,
        x1=x[0],
        x2=x[1],
    )


@st.composite
def dims_and_offset(draw, shape):
    shape_actual = draw(shape)
    dim1 = draw(helpers.get_axis(shape=shape, force_int=True))
    dim2 = draw(helpers.get_axis(shape=shape, force_int=True))
    offset = draw(
        st.integers(min_value=-shape_actual[dim1], max_value=shape_actual[dim1])
    )
    return dim1, dim2, offset


# diagonal
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


# expand_dims
@handle_frontend_test(
    fn_tree="jax.numpy.expand_dims",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(), key="expand_dims_axis"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="expand_dims_axis"),
    ),
)
def test_jax_expand_dims(
        *,
        dtype_and_x,
        axis,
        with_out,
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
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
    )


# degrees
@handle_frontend_test(
    fn_tree="jax.numpy.degrees",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_numpy_degrees(
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


# asarray
@handle_frontend_test(
    fn_tree="jax.numpy.asarray",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
)
def test_jax_numpy_asarray(
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
        a=a,
        dtype=dtype[0],
    )


# take
@handle_frontend_test(
    fn_tree="jax.numpy.take",
    dtype_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=helpers.get_dtypes("integer"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
    ),
)
def test_jax_numpy_take(
        *,
        dtype_indices_axis,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtypes, value, indices, axis, _ = dtype_indices_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=3,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=value,
        indices=indices,
        axis=axis,
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


# stack
@handle_frontend_test(
    fn_tree="jax.numpy.stack",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays"),
        shape=helpers.get_shape(min_num_dims=1),
        shared_dtype=True,
        valid_axis=True,
        allow_neg_axes=True,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_jax_numpy_stack(
        dtype_values_axis,
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        on_device,
        fn_tree,
        frontend,
):
    input_dtype, values, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        arrays=values,
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.negative",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1
    ),
)
def test_jax_numpy_negative(
        dtype_and_x,
        as_variable,
        with_out,
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
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.rad2deg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1
    ),
)
def test_jax_numpy_rad2deg(
        dtype_and_x,
        as_variable,
        with_out,
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
        x=x[0],
    )


# tensordot
@handle_frontend_test(
    fn_tree="jax.numpy.tensordot",
    dtype_values_and_axes=_get_dtype_value1_value2_axis_for_tensordot(
        helpers.get_dtypes(kind="numeric")
    ),
)
def test_jax_numpy_tensordot(
    dtype_values_and_axes,
    as_variable,
    native_array,
    num_positional_args,
    frontend,
    fn_tree,
):
    dtype, a, b, axes = dtype_values_and_axes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        a=a,
        b=b,
        axes=axes,
    )
