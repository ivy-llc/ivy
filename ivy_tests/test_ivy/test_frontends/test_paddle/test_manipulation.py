# global
from hypothesis import strategies as st
import math

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_manipulation import (  # noqa
    _get_dtype_values_k_axes_for_rot90,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_miscellaneous_ops import (
    _get_repeat_interleaves_args,
)


# --- Helpers --- #
# --------------- #


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


@st.composite
def _arrays_dim_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = 2
    common_shape = draw(
        helpers.lists(
            x=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    _dim = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            x=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )

    min_dim = min(unique_dims)
    max_dim = max(unique_dims)
    _idx = draw(
        helpers.array_values(
            shape=min_dim,
            dtype="int64",
            min_value=0,
            max_value=max_dim,
            exclude_min=False,
        )
    )

    xs = []
    # available_input_types = draw(helpers.get_dtypes("integer"))
    available_input_types = ["int32", "int64", "float16", "float32", "float64"]
    input_dtypes = draw(
        helpers.array_dtypes(
            available_dtypes=available_input_types,
            num_arrays=num_arrays,
            shared_dtype=True,
        )
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:_dim] + [ud] + common_shape[_dim:],
                dtype=dt,
                large_abs_safety_factor=2.5,
                small_abs_safety_factor=2.5,
                safety_factor_scale="log",
            )
        )
        xs.append(x)
    return xs, input_dtypes, _dim, _idx


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


@st.composite
def _broadcast_to_helper(draw):
    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=6,
        )
    )

    dtype, x = dtype_and_x
    input_shape = x[0].shape

    max_num_dims = 6 - len(input_shape)
    shape = draw(helpers.get_shape(max_num_dims=max_num_dims)) + input_shape

    return dtype, x, shape


# flip
@st.composite
def _dtype_x_axis(draw, **kwargs):
    dtype, x, shape = draw(helpers.dtype_and_values(**kwargs, ret_shape=True))
    axis = draw(
        st.lists(
            helpers.ints(min_value=0, max_value=len(shape) - 1),
            min_size=len(shape),
            max_size=len(shape),
            unique=True,
        )
    )
    return dtype, x, axis


# expand
@st.composite
def _expand_helper(draw):
    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=6,
        )
    )

    dtype, x = dtype_and_x
    input_shape = x[0].shape

    max_num_dims = 6 - len(input_shape)
    shape = draw(helpers.get_shape(max_num_dims=max_num_dims)) + input_shape

    return dtype, x, shape


@st.composite
def _gather_helper(draw):
    dtype_and_param = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=6,
        )
    )

    dtype_and_indices = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=6,
        )
    )
    dtype, param = dtype_and_param
    dtype, indices = dtype_and_indices
    return dtype, param, indices


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


# --- Main --- #
# ------------ #


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
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="paddle.broadcast_to",
    dtype_x_and_shape=_broadcast_to_helper(),
)
def test_paddle_broadcast_to(
    *,
    dtype_x_and_shape,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, x, shape = dtype_x_and_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        shape=shape,
    )


# cast
@handle_frontend_test(
    fn_tree="paddle.cast",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_paddle_cast(
    *,
    dtype_and_x,
    dtype,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        dtype=dtype[0],
    )


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
    backend_fw,
    test_flags,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs,
        axis=unique_idx,
    )


@handle_frontend_test(
    fn_tree="paddle.expand",
    dtype_x_and_shape=_expand_helper(),
)
def test_paddle_expand(
    *,
    dtype_x_and_shape,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, x, shape = dtype_x_and_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        shape=shape,
    )


@handle_frontend_test(
    fn_tree="paddle.flip",
    dtype_x_axis=_dtype_x_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    test_with_out=st.just(False),
)
def test_paddle_flip(
    *,
    dtype_x_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="paddle.gather",
    dtype_param_and_indices=_gather_helper(),
)
def test_paddle_gather(
    *,
    dtype_param_and_indices,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, param, indices = dtype_param_and_indices
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        param=param[0],
        indices=indices[0],
    )


# gather_nd
@handle_frontend_test(
    fn_tree="paddle.gather_nd",
    dtype_x_index=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("valid"),
        indices_dtypes=["int64"],
        min_num_dims=5,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=5,
        indices_same_dims=False,
    ),
)
def test_paddle_gather_nd(
    *,
    dtype_x_index,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, index, _, _ = dtype_x_index
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        index=index,
    )


@handle_frontend_test(
    fn_tree="paddle.index_add",
    xs_dtypes_dim_idx=_arrays_dim_idx_n_dtypes(),
)
def test_paddle_index_add(
    *,
    xs_dtypes_dim_idx,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    xs, input_dtypes, axis, indices = xs_dtypes_dim_idx
    if xs[0].shape[axis] < xs[1].shape[axis]:
        source, input = xs
    else:
        input, source = xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        x=input,
        index=indices,
        axis=axis,
        value=source,
    )


# repeat_interleave
@handle_frontend_test(
    fn_tree="paddle.repeat_interleave",
    dtype_values_repeats_axis_output_size=_get_repeat_interleaves_args(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        max_num_dims=4,
        max_dim_size=4,
    ),
)
def test_paddle_repeat_interleave(
    *,
    dtype_values_repeats_axis_output_size,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, values, repeats, axis, _ = dtype_values_repeats_axis_output_size

    helpers.test_frontend_function(
        input_dtypes=[dtype[0][0], dtype[1][0]],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=values[0],
        repeats=repeats[0],
        axis=axis,
    )


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
    backend_fw,
):
    input_dtype, x, shape = dtypes_x_reshape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        shape=shape,
    )


# roll
@handle_frontend_test(
    fn_tree="paddle.roll",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    shift=helpers.ints(min_value=1, max_value=10),
    axis=helpers.ints(min_value=-1, max_value=1),
    test_with_out=st.just(False),
)
def test_paddle_roll(
    *,
    dtype_and_x,
    shift,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        shifts=shift,
        axis=axis,
    )


# rot90
@handle_frontend_test(
    fn_tree="paddle.rot90",
    dtype_m_k_axes=_get_dtype_values_k_axes_for_rot90(
        available_dtypes=helpers.get_dtypes(kind="valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_paddle_rot90(
    *,
    dtype_m_k_axes,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, m, k, axes = dtype_m_k_axes
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=m,
        k=k,
        axes=tuple(axes),
    )


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
    backend_fw,
):
    input_dtypes, x, num_splits, axis = dt_x_num_splits_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        num_or_sections=num_splits,
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="paddle.squeeze",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=_squeeze_helper(),
)
def test_paddle_squeeze(
    *,
    dtype_and_x,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
    )


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
    backend_fw,
):
    xs, input_dtypes, axis = _arrays_n_dtypes_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs,
        axis=axis,
    )


# take_along_axis
@handle_frontend_test(
    fn_tree="paddle.take_along_axis",
    dtype_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes(kind="valid"),
        indices_dtypes=["int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
    ),
)
def test_paddle_take_along_axis(
    *,
    dtype_indices_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtypes, value, indices, axis, _ = dtype_indices_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=value,
        indices=indices,
        axis=axis,
    )


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
    backend_fw,
    test_flags,
):
    input_dtypes, x, repeats = dt_x_repeats
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        repeat_times=repeats,
    )


@handle_frontend_test(
    fn_tree="paddle.tolist",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    test_with_out=st.just(False),
)
def test_paddle_tolist(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    x_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# unbind
@handle_frontend_test(
    fn_tree="paddle.unbind",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        max_num_dims=2,
        max_dim_size=1,
    ),
    number_positional_args=st.just(1),
    axis=st.integers(-1, 0),
    test_with_out=st.just(False),
)
def test_paddle_unbind(
    *,
    dtypes_values,
    axis,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    x_dtype, x = dtypes_values
    axis = axis
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
    )


# unstack
@handle_frontend_test(
    fn_tree="paddle.unstack",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        max_dim_size=1,
    ),
    number_positional_args=st.just(1),
    axis=st.integers(-1, 0),
    test_with_out=st.just(False),
)
def test_paddle_unstack(
    *,
    dtypes_values,
    axis,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    x_dtype, x = dtypes_values
    axis = axis
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
    )
