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
    test_with_out=st.just(False),
)
def test_numpy_where(
    broadcastables,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    cond, x1, x2, dtype = broadcastables
    helpers.test_frontend_function(
        input_dtypes=["bool", dtype],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_numpy_nonzero(
    dtype_and_a,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_numpy_argmin(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
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
    test_with_out=st.just(False),
)
def test_numpy_argmax(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


# flatnonzero
@handle_frontend_test(
    fn_tree="numpy.flatnonzero",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_numpy_flatnonzero(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# searchsorted
@st.composite
def _search_sorted_values(draw):
    case = st.booleans()
    if case:
        # when x is 1-D and v is N-D
        dtype_x, x = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes(
                    "numeric", full=False, key="searchsorted"
                ),
                shape=(draw(st.integers(min_value=1, max_value=5)),),
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
    else:
        # when x is N-D and v is N-D
        lead_dim = draw(
            helpers.get_shape(min_num_dims=1),
        )
        nx = draw(st.integers(min_value=1, max_value=5))
        nv = draw(st.integers(min_value=1, max_value=5))
        dtype_x, x = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes(
                    "numeric", full=False, key="searchsorted"
                ),
                shape=lead_dim + (nx,),
            ),
        )
        dtype_v, v = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes(
                    "numeric", full=False, key="searchsorted"
                ),
                shape=lead_dim + (nv,),
            ),
        )
    input_dtypes = dtype_x + dtype_v
    xs = x + v
    side = draw(st.sampled_from(["left", "right"]))
    use_sorter = draw(st.booleans())
    if use_sorter:
        sorter_dtype = draw(st.sampled_from(["int32", "int64"]))
        input_dtypes.append(sorter_dtype)
        sorter = np.argsort(xs[0], axis=-1).astype(sorter_dtype)
    else:
        sorter = None
        xs[0] = np.sort(xs[0], axis=-1)
    return input_dtypes, xs, side, sorter


@handle_frontend_test(
    fn_tree="numpy.searchsorted",
    dtype_x_v_side_sorter=_search_sorted_values(),
    test_with_out=st.just(False),
)
def test_numpy_searchsorted(
    dtype_x_v_side_sorter,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs, side, sorter = dtype_x_v_side_sorter
    helpers.test_frontend_function(
        input_dtypes=input_dtypes + ["int64"],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        v=xs[1],
        side=side,
        sorter=sorter,
    )


# argwhere
@handle_frontend_test(
    fn_tree="numpy.argwhere",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_with_out=st.just(False),
)
def test_numpy_argwhere(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_numpy_nanargmax(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    fn_tree="numpy.nanargmin",
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
def test_numpy_nanargmin(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


@st.composite
def _extract_strategy(draw):
    dtype_and_cond = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
        )
    )
    dtype_and_arr = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
        )
    )
    return dtype_and_cond, dtype_and_arr


# extract
@handle_frontend_test(
    fn_tree="numpy.extract",
    dtype_and_x=_extract_strategy(),
    test_with_out=st.just(False),
)
def test_numpy_extract(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype_cond, cond = dtype_and_x[0]
    dtype_arr, arr = dtype_and_x[1]

    helpers.test_frontend_function(
        input_dtypes=dtype_cond + dtype_arr,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        cond=cond[0],
        arr=arr[0],
    )
