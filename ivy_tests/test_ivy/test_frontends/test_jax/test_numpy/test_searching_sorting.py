# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_searching import (
    _broadcastable_trio,
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
def test_jax_argmax(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_argwhere(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_argsort(
    *,
    dtype_x_axis,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
    )


# msort
# @handle_frontend_test(
#     fn_tree="jax.numpy.msort",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("numeric"),
#         min_num_dims=2,
#         min_dim_size=2,
#     ),
#     test_with_out=st.just(False),
# )
# def test_jax_msort(
#     dtype_and_x,
#     frontend,
#     test_flags,
#     fn_tree,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         a=x[0],
#     )
# TODO : deprecated since jax 0.4.1. Uncomment with multiversion testing pipeline
# enabled.


# nonzero
@handle_frontend_test(
    fn_tree="jax.numpy.nonzero",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_with_out=st.just(False),
)
def test_jax_nonzero(
    dtype_and_a,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
    )


# nanargmax
@handle_frontend_test(
    fn_tree="jax.numpy.nanargmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_nanargmax(
    dtype_x_axis,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


# nanargmin
@handle_frontend_test(
    fn_tree="jax.numpy.nanargmin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_nanargmin(
    dtype_x_axis,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


# extract
@handle_frontend_test(
    fn_tree="jax.numpy.extract",
    broadcastables=_broadcastable_trio(),
)
def test_jax_extract(
    broadcastables,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    cond, xs, dtype = broadcastables
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        condition=cond,
        arr=xs[0],
    )


# sort
@handle_frontend_test(
    fn_tree="jax.numpy.sort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_sort(
    *,
    dtype_x_axis,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
    )


# flatnonzero
@handle_frontend_test(
    fn_tree="jax.numpy.flatnonzero",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_with_out=st.just(False),
)
def test_jax_flatnonzero(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# sort_complex
@handle_frontend_test(
    fn_tree="jax.numpy.sort_complex",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        min_axis=-1,
        max_axis=0,
    ),
    test_with_out=st.just(False),
)
def test_jax_sort_complex(
    *,
    dtype_x_axis,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        test_values=False,
    )


# searchsorted
@st.composite
def _searchsorted(draw):
    dtype_x, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "numeric", full=False, key="searchsorted"
            ),
            shape=(draw(st.integers(min_value=1, max_value=10)),),
        ),
    )
    dtype_v, v = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "numeric", full=False, key="searchsorted"
            ),
            min_num_dims=1,
        )
    )

    input_dtypes = dtype_x + dtype_v
    xs = x + v
    side = draw(st.sampled_from(["left", "right"]))
    sorter = None
    xs[0] = np.sort(xs[0], axis=-1)
    return input_dtypes, xs, side, sorter


@handle_frontend_test(
    fn_tree="jax.numpy.searchsorted",
    dtype_x_v_side_sorter=_searchsorted(),
    test_with_out=st.just(False),
)
def test_numpy_searchsorted(
    dtype_x_v_side_sorter,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, xs, side, sorter = dtype_x_v_side_sorter
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        v=xs[1],
        side=side,
        sorter=sorter,
    )


# where
@handle_frontend_test(
    fn_tree="jax.numpy.where",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_jax_where(
    *,
    dtype_and_x,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
    test_flags,
):
    input_dtype, x = dtype_and_x
    x = x[0]
    condition = x > 0.5
    x1 = x * 2
    x2 = x * -2

    # Convert input_dtype from list to string
    input_dtype = input_dtype[0]

    helpers.test_frontend_function(
        input_dtypes=["bool", input_dtype, input_dtype],
        fn_tree=fn_tree,
        on_device=on_device,
        test_flags=test_flags,
        frontend=frontend,
        condition=condition,
        x=x1,
        y=x2,
    )


# unique
@st.composite
def _unique_helper(draw):
    arr_dtype, arr, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "numeric", full=False, key="searchsorted"
            ),
            min_num_dims=1,
            min_dim_size=2,
            ret_shape=True,
        )
    )
    axis = draw(st.sampled_from(list(range(len(shape))) + [None]))
    return_index = draw(st.booleans())
    return_inverse = draw(st.booleans())
    return_counts = draw(st.booleans())
    return arr_dtype, arr, return_index, return_inverse, return_counts, axis


@handle_frontend_test(
    fn_tree="jax.numpy.unique", fn_inputs=_unique_helper(), test_with_out=st.just(False)
)
def test_jax_unique(fn_inputs, backend_fw, frontend, test_flags, fn_tree, on_device):
    arr_dtype, arr, return_index, return_inverse, return_counts, axis = fn_inputs
    helpers.test_frontend_function(
        input_dtypes=arr_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ar=arr[0],
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    )
