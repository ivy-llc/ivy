# global
from hypothesis import strategies as st
import math

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Helpers #
# ------ #


@st.composite
def dtypes_x_reshape(draw):
    shape = draw(helpers.get_shape(min_num_dims=1))
    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    shape = draw(
        helpers.get_shape(min_num_dims=1).filter(
            lambda s: math.prod(s) == math.prod(shape)
        )
    )
    return dtypes, x, shape


# Tests #
# ----- #


# reshape
@handle_frontend_test(
    fn_tree="paddle.reshape",
    dtypes_x_reshape=dtypes_x_reshape(),
)
def test_paddle_reshape(
    *,
    dtypes_x_reshape,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, shape = dtypes_x_reshape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        shape=shape,
    )


# abs
@handle_frontend_test(
    fn_tree="paddle.abs",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_abs(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# stack
@st.composite
def _arrays_axis_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=2, max_value=5), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.list_of_size(
            x=helpers.ints(min_value=2, max_value=3),
            size=num_dims - 1,
        )
    )
    axis = draw(st.sampled_from(list(range(num_dims))))
    xs = []
    input_dtypes = draw(
        helpers.array_dtypes(available_dtypes=draw(helpers.get_dtypes("numeric")))
    )
    dtype = draw(st.sampled_from(input_dtypes))
    for _ in range(num_arrays):
        x = draw(
            helpers.array_values(
                shape=common_shape,
                dtype=dtype,
            )
        )
        xs.append(x)
    input_dtypes = [dtype] * len(input_dtypes)
    return xs, input_dtypes, axis


@handle_frontend_test(
    fn_tree="paddle.stack",
    _arrays_n_dtypes_axis=_arrays_axis_n_dtypes(),
    test_with_out=st.just(False),
)
def test_paddle_stack(
    *,
    _arrays_n_dtypes_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    xs, input_dtypes, axis = _arrays_n_dtypes_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs,
        axis=axis,
    )


# concat
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.list_of_size(
            x=helpers.ints(min_value=2, max_value=3),
            size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.list_of_size(
            x=helpers.ints(min_value=2, max_value=3),
            size=num_arrays,
        )
    )
    xs = []
    input_dtypes = draw(
        helpers.array_dtypes(available_dtypes=draw(helpers.get_dtypes("valid")))
    )
    dtype = draw(st.sampled_from(input_dtypes))
    for ud in unique_dims:
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dtype,
            )
        )
        xs.append(x)
    input_dtypes = [dtype] * len(input_dtypes)
    return xs, input_dtypes, unique_idx


@handle_frontend_test(
    fn_tree="paddle.concat",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    test_with_out=st.just(False),
)
def test_paddle_concat(
    *,
    xs_n_input_dtypes_n_unique_idx,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs,
        axis=unique_idx,
    )


# tile
@st.composite
def _tile_helper(draw):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=4,
            min_dim_size=2,
            max_dim_size=3,
            ret_shape=True,
        )
    )
    repeats = draw(
        helpers.list_of_size(
            x=helpers.ints(min_value=1, max_value=3),
            size=len(shape),
        )
    )
    return dtype, x, repeats


@handle_frontend_test(
    fn_tree="paddle.tile",
    dt_x_repeats=_tile_helper(),
    test_with_out=st.just(False),
)
def test_paddle_tile(
    *,
    dt_x_repeats,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, repeats = dt_x_repeats
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        repeat_times=repeats,
    )


# split
@st.composite
def _split_helper(draw):
    dtypes, values, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=2,
            max_num_dims=4,
            min_dim_size=2,
            max_dim_size=4,
            ret_shape=True,
        )
    )
    axis = draw(st.sampled_from(range(len(shape))))
    num_eles = shape[axis]
    splits = [i for i in range(1, num_eles + 1) if num_eles % i == 0]
    num_splits = draw(st.sampled_from(splits))
    return dtypes, values, num_splits, axis


@handle_frontend_test(
    fn_tree="paddle.split",
    dt_x_num_splits_axis=_split_helper(),
    test_with_out=st.just(False),
)
def test_paddle_split(
    *,
    dt_x_num_splits_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, num_splits, axis = dt_x_num_splits_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        num_or_sections=num_splits,
        axis=axis,
    )
