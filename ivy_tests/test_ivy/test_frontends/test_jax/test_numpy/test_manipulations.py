# global
from hypothesis import strategies as st, assume
import numpy as np
import hypothesis.extra.numpy as nph

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import (
    _repeat_helper,
)
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_manipulation import (  # noqa
    _get_dtype_values_k_axes_for_rot90,
    _get_splits,
    _st_tuples_or_int,
)


# --- Helpers --- #
# --------------- #


# concatenate
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


# block
@st.composite
def _get_input_and_block(draw):
    shapes = draw(
        st.lists(
            helpers.get_shape(
                min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
            ),
            min_size=2,
            max_size=10,
        )
    )
    x_dtypes, xs = zip(
        *[
            draw(
                helpers.dtype_and_values(
                    available_dtypes=helpers.get_dtypes("valid"),
                    min_num_dims=1,
                    max_num_dims=5,
                    min_dim_size=2,
                    max_dim_size=10,
                    shape=shape,
                )
            )
            for shape in shapes
        ]
    )
    return x_dtypes, xs


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


# resize
@st.composite
def _get_input_and_new_shape(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    new_shape = draw(
        helpers.get_shape(
            min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=10
        ).filter(lambda x: np.prod(x) == np.prod(shape))
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=2,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=10,
            shape=shape,
        )
    )
    return x_dtype, x, new_shape


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


# swapaxes
@st.composite
def _get_input_and_two_swapabble_axes(draw):
    x_dtype, x, x_shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            ret_shape=True,
            min_num_dims=1,
            max_num_dims=10,
        )
    )

    axis1 = draw(
        helpers.ints(
            min_value=-1 * len(x_shape),
            max_value=len(x_shape) - 1,
        )
    )
    axis2 = draw(
        helpers.ints(
            min_value=-1 * len(x_shape),
            max_value=len(x_shape) - 1,
        )
    )
    return x_dtype, x, axis1, axis2


# pad
@st.composite
def _pad_helper(draw):
    mode = draw(
        st.sampled_from(
            [
                "constant",
                "edge",
                "linear_ramp",
                "maximum",
                "mean",
                "median",
                "minimum",
                "reflect",
                "symmetric",
                "wrap",
            ]
        )
    )
    if mode == "median":
        dtypes = "float"
    else:
        dtypes = "numeric"
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(dtypes),
            ret_shape=True,
            min_num_dims=1,
            min_value=-100,
            max_value=100,
        ).filter(
            lambda x: x[0][0] not in ["float16", "bfloat16", "complex64", "complex128"]
        ),
    )
    ndim = len(shape)
    pad_width = draw(_st_tuples_or_int(ndim, min_val=0))
    kwargs = {}
    if mode in ["reflect", "symmetric"]:
        kwargs["reflect_type"] = draw(st.sampled_from(["even", "odd"]))
    if mode in ["maximum", "mean", "median", "minimum"]:
        kwargs["stat_length"] = draw(_st_tuples_or_int(ndim, min_val=2))
    if mode in ["linear_ramp"]:
        kwargs["end_values"] = draw(_st_tuples_or_int(ndim))
    if mode == "constant":
        kwargs["constant_values"] = draw(_st_tuples_or_int(ndim))
    return dtype, input[0], pad_width, kwargs, mode


# TODO: uncomment when block is reimplemented
# @handle_frontend_test(
#     fn_tree="jax.numpy.block",
#     input_x_shape=_get_input_and_block(),
#     test_with_out=st.just(False),
# )
# def test_jax_block(
#     *,
#     input_x_shape,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     x_dtypes, xs = input_x_shape
#     helpers.test_frontend_function(
#         input_dtypes=x_dtypes,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         arrays=xs,
#     )


@st.composite
def _squeeze_helper(draw):
    shape = draw(st.shared(helpers.get_shape(), key="shape"))
    valid_axes = [idx for idx in range(len(shape)) if shape[idx] == 1] + [None]
    return draw(st.sampled_from(valid_axes))


# --- Main --- #
# ------------ #


# append
@handle_frontend_test(
    fn_tree="jax.numpy.append",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shape=helpers.get_shape(
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
        ),
        shared_dtype=True,
        valid_axis=True,
        allow_neg_axes=True,
        force_int_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_append(
    dtype_values_axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, values, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=values[0],
        values=values[1],
        axis=axis,
    )


# array_split
@handle_frontend_test(
    fn_tree="jax.numpy.array_split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=1, allow_none=False, is_mod_split=True
    ),
    axis=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    test_with_out=st.just(False),
)
def test_jax_array_split(
    *,
    dtype_value,
    indices_or_sections,
    axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, value = dtype_value
    assume(isinstance(indices_or_sections, int))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
        axis=axis,
    )


# atleast_1d
@handle_frontend_test(
    fn_tree="jax.numpy.atleast_1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_jax_atleast_1d(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys[f"arrs{i}"] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
    )


# atleast_2d
@handle_frontend_test(
    fn_tree="jax.numpy.atleast_2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_jax_atleast_2d(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys[f"arrs{i}"] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
    )


# atleast_3d
@handle_frontend_test(
    fn_tree="jax.numpy.atleast_3d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_jax_atleast_3d(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys[f"arrs{i}"] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
    )


# bartlett
@handle_frontend_test(
    fn_tree="jax.numpy.bartlett",
    m=helpers.ints(min_value=0, max_value=20),
)
def test_jax_bartlett(
    m,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=["int64"],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        M=m,
    )


# blackman
@handle_frontend_test(
    fn_tree="jax.numpy.blackman",
    m=helpers.ints(min_value=0, max_value=20),
)
def test_jax_blackman(
    m,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=["int64"],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        M=m,
    )


# broadcast_arrays
@handle_frontend_test(
    fn_tree="jax.numpy.broadcast_arrays",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_broadcast_arrays(
    *,
    dtype_value,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, value = dtype_value
    arrys = {}
    for i, v in enumerate(value):
        arrys[f"array{i}"] = v
    test_flags.num_positional_args = len(arrys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arrys,
    )


# broadcast_shapes
@handle_frontend_test(
    fn_tree="jax.numpy.broadcast_shapes",
    shapes=nph.mutually_broadcastable_shapes(
        num_shapes=4, min_dims=1, max_dims=5, min_side=1, max_side=5
    ),
    test_with_out=st.just(False),
)
def test_jax_broadcast_shapes(
    *,
    shapes,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    shape, _ = shapes
    shapes = {f"shape{i}": shape[i] for i in range(len(shape))}
    test_flags.num_positional_args = len(shapes)
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=["int64"],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **shapes,
        test_values=False,
    )
    assert ret == frontend_ret


@handle_frontend_test(
    fn_tree="jax.numpy.broadcast_to",
    input_x_broadcast=_get_input_and_broadcast_shape(),
    test_with_out=st.just(False),
)
def test_jax_broadcast_to(
    *,
    input_x_broadcast,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    x_dtype, x, shape = input_x_broadcast
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=x[0],
        shape=shape,
    )


# clip
@handle_frontend_test(
    fn_tree="jax.numpy.clip",
    input_and_ranges=_get_clip_inputs(),
)
def test_jax_clip(
    *,
    input_and_ranges,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    x_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        a_min=min,
        a_max=max,
    )


# column_stack
@handle_frontend_test(
    fn_tree="jax.numpy.column_stack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    factor=helpers.ints(min_value=2, max_value=6),
)
def test_jax_column_stack(
    dtype_and_x,
    factor,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    ys = [x[0]]
    for i in range(factor):
        ys += [x[0]]
    helpers.test_frontend_function(
        input_dtypes=[dtype[0]] * (factor + 1),
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tup=ys,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.concatenate",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    test_with_out=st.just(False),
)
def test_jax_concat(
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
        arrays=xs,
        axis=unique_idx,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.diagflat",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=helpers.get_shape(
            min_num_dims=1, max_num_dims=2, min_dim_size=1, max_dim_size=10
        ),
        small_abs_safety_factor=2.5,
        large_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    k=st.integers(min_value=-5, max_value=5),
)
def test_jax_diagflat(
    dtype_x,
    k,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        v=x[0],
        k=k,
    )


# dsplit
@handle_frontend_test(
    fn_tree="jax.numpy.dsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=3), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=3, axis=2, allow_none=False, is_mod_split=True
    ),
    test_with_out=st.just(False),
)
def test_jax_dsplit(
    *,
    dtype_value,
    indices_or_sections,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
    )


# dstack
@handle_frontend_test(
    fn_tree="jax.numpy.dstack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shared_dtype=True,
        num_arrays=helpers.ints(min_value=1, max_value=10),
        shape=helpers.get_shape(
            min_num_dims=1,
        ),
    ),
    test_with_out=st.just(False),
)
def test_jax_dstack(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
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
        tup=x,
    )


# expand_dims
@handle_frontend_test(
    fn_tree="jax.numpy.expand_dims",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        force_int_axis=True,
        valid_axis=True,
    ),
)
def test_jax_expand_dims(
    *,
    dtype_x_axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
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
    )


# flip
@handle_frontend_test(
    fn_tree="jax.numpy.flip",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_flip(
    *,
    dtype_value,
    axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_fliplr(
    *,
    dtype_and_m,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, m = dtype_and_m
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=m[0],
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
    test_with_out=st.just(False),
)
def test_jax_flipud(
    *,
    dtype_and_m,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, m = dtype_and_m
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=m[0],
    )


# hamming
@handle_frontend_test(
    fn_tree="jax.numpy.hamming",
    m=helpers.ints(min_value=0, max_value=20),
)
def test_jax_hamming(
    m,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=["int64"],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        M=m,
    )


# hanning
@handle_frontend_test(
    fn_tree="jax.numpy.hanning",
    m=helpers.ints(min_value=0, max_value=20),
)
def test_jax_hanning(
    m,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=["int64"],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        M=m,
    )


# hsplit
@handle_frontend_test(
    fn_tree="jax.numpy.hsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=2, axis=1, allow_none=False, is_mod_split=True
    ),
    test_with_out=st.just(False),
)
def test_jax_hsplit(
    *,
    dtype_value,
    indices_or_sections,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
    )


# kaiser
@handle_frontend_test(
    fn_tree="jax.numpy.kaiser",
    m=helpers.ints(min_value=0, max_value=100),
    beta=helpers.floats(min_value=-10, max_value=10),
)
def test_jax_kaiser(
    m,
    beta,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=["int64", "float64"],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        M=m,
        beta=beta,
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
    test_with_out=st.just(False),
)
def test_jax_moveaxis(
    *,
    dtype_and_a,
    source,
    destination,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
        source=source,
        destination=destination,
    )


# trim_zeros
@handle_frontend_test(
    fn_tree="jax.numpy.trim_zeros",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1, max_num_dims=1
    ),
    trim=st.sampled_from(["f", "b", "fb"]),
)
def test_jax_numpy_trim_zeros(
    frontend,
    on_device,
    *,
    dtype_and_x,
    backend_fw,
    trim,
    fn_tree,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        filt=x[0],
        trim=trim,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.pad",
    dtype_and_input_and_other=_pad_helper(),
    test_with_out=st.just(False),
)
def test_jax_pad(
    *,
    dtype_and_input_and_other,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    (
        dtype,
        input,
        pad_width,
        kwargs,
        mode,
    ) = dtype_and_input_and_other

    if isinstance(pad_width, int):
        pad_width = ((pad_width, pad_width),) * input.ndim
    else:
        pad_width = tuple(
            tuple(pair) if isinstance(pair, list) else pair for pair in pad_width
        )

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=input,
        pad_width=pad_width,
        mode=mode,
        **kwargs,
    )


# ravel
@handle_frontend_test(
    fn_tree="jax.numpy.ravel",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        shape=helpers.get_shape(
            min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=10
        ),
    ),
    order=st.sampled_from(["C", "F"]),
    test_with_out=st.just(False),
)
def test_jax_ravel(
    *,
    dtype_and_values,
    order,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        order=order,
    )


# repeat
@handle_frontend_test(
    fn_tree="jax.numpy.repeat",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    axis=st.shared(
        st.one_of(
            st.none(),
            helpers.get_axis(
                shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
                max_size=1,
            ),
        ),
        key="axis",
    ),
    repeat=st.one_of(st.integers(1, 10), _repeat_helper()),
    test_with_out=st.just(False),
)
def test_jax_repeat(
    *,
    dtype_value,
    axis,
    repeat,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    value_dtype, value = dtype_value

    if not isinstance(repeat, int):
        repeat_dtype, repeat_list = repeat
        repeat = repeat_list[0]
        value_dtype += repeat_dtype

    if not isinstance(axis, int) and axis is not None:
        axis = axis[0]

    helpers.test_frontend_function(
        input_dtypes=value_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=value[0],
        repeats=repeat,
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.reshape",
    input_x_shape=_get_input_and_reshape(),
    order=st.sampled_from(["C", "F"]),
    test_with_out=st.just(False),
)
def test_jax_reshape(
    *,
    input_x_shape,
    order,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    x_dtype, x, shape = input_x_shape
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        newshape=shape,
        order=order,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.resize",
    input_x_shape=_get_input_and_new_shape(),
    test_with_out=st.just(True),
)
def test_jax_resize(
    *,
    input_x_shape,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    x_dtype, x, new_shape = input_x_shape
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        new_shape=new_shape,
    )


# roll
@handle_frontend_test(
    fn_tree="jax.numpy.roll",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    shift=helpers.dtype_and_values(
        available_dtypes=[ivy.int32],
        max_num_dims=1,
        min_dim_size=st.shared(
            helpers.ints(min_value=1, max_value=10),
            key="shift_len",
        ),
        max_dim_size=st.shared(
            helpers.ints(min_value=1, max_value=10),
            key="shift_len",
        ),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
        force_tuple=True,
        unique=False,
        min_size=st.shared(
            helpers.ints(min_value=1, max_value=10),
            key="shift_len",
        ),
        max_size=st.shared(
            helpers.ints(min_value=1, max_value=10),
            key="shift_len",
        ),
    ),
    test_with_out=st.just(False),
)
def test_jax_roll(
    *,
    dtype_value,
    shift,
    axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    value_dtype, value = dtype_value
    shift_dtype, shift_val = shift

    if shift_val[0].ndim == 0:  # If shift is an int
        shift_val = shift_val[0]  # Drop shift's dtype (always int32)
        axis = axis[0]  # Extract an axis value from the tuple
    else:
        # Drop shift's dtype (always int32) and convert list to tuple
        shift_val = tuple(shift_val[0].tolist())

    helpers.test_frontend_function(
        input_dtypes=value_dtype + shift_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=value[0],
        shift=shift_val,
        axis=axis,
    )


# rot90
@handle_frontend_test(
    fn_tree="jax.numpy.rot90",
    dtype_m_k_axes=_get_dtype_values_k_axes_for_rot90(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    test_with_out=st.just(False),
)
def test_jax_rot90(
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
        m=m,
        k=k,
        axes=tuple(axes),
    )


# row_stack
@handle_frontend_test(
    fn_tree="jax.numpy.row_stack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    factor=helpers.ints(min_value=2, max_value=6),
)
def test_jax_row_stack(
    dtype_and_x,
    factor,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    xs = [x[0]]
    for i in range(factor):
        xs += [x[0]]
    helpers.test_frontend_function(
        input_dtypes=[dtype[0]] * (factor + 1),
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tup=xs,
    )


# split
@handle_frontend_test(
    fn_tree="jax.numpy.split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=1, allow_none=False, is_mod_split=True
    ),
    axis=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    test_with_out=st.just(False),
)
def test_jax_split(
    *,
    dtype_value,
    indices_or_sections,
    axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
        axis=axis,
    )


# squeeze
@handle_frontend_test(
    fn_tree="jax.numpy.squeeze",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    axis=_squeeze_helper(),
    test_with_out=st.just(False),
)
def test_jax_squeeze(
    *,
    dtype_and_values,
    axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=values[0],
        axis=axis,
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
def test_jax_stack(
    dtype_values_axis,
    dtype,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, values, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        arrays=values,
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.swapaxes",
    input_x_axis1_axis2=_get_input_and_two_swapabble_axes(),
    test_with_out=st.just(False),
)
def test_jax_swapaxes(
    *,
    input_x_axis1_axis2,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    x_dtype, x, axis1, axis2 = input_x_axis1_axis2
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis1=axis1,
        axis2=axis2,
    )


# take
@handle_frontend_test(
    fn_tree="jax.numpy.take",
    dtype_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
    ),
)
def test_jax_take(
    *,
    dtype_indices_axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtypes, value, indices, axis, _ = dtype_indices_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=value,
        indices=indices,
        axis=axis,
    )


# tile
@handle_frontend_test(
    fn_tree="jax.numpy.tile",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    repeat=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape").map(
            lambda rep: (len(rep),)
        ),
        min_value=0,
        max_value=10,
    ),
    test_with_out=st.just(False),
)
def test_jax_tile(
    *,
    dtype_value,
    repeat,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, value = dtype_value
    repeat_dtype, repeat_list = repeat
    helpers.test_frontend_function(
        input_dtypes=dtype + repeat_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        A=value[0],
        reps=repeat_list[0],
    )


# transpose
@handle_frontend_test(
    fn_tree="jax.numpy.transpose",
    array_and_axes=np_frontend_helpers._array_and_axes_permute_helper(
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=0,
        max_dim_size=10,
    ),
    test_with_out=st.just(False),
)
def test_jax_transpose(
    *,
    array_and_axes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    array, dtype, axes = array_and_axes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=array,
        axes=axes,
    )


# tri
@handle_frontend_test(
    fn_tree="jax.numpy.tri",
    rows=helpers.ints(min_value=3, max_value=10),
    cols=helpers.ints(min_value=3, max_value=10),
    k=helpers.ints(min_value=-10, max_value=10),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_tri(
    rows,
    cols,
    k,
    dtype,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        N=rows,
        M=cols,
        k=k,
        dtype=dtype[0],
    )


# tril
@handle_frontend_test(
    fn_tree="jax.numpy.tril",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    test_with_out=st.just(False),
)
def test_jax_tril(
    *,
    dtype_and_x,
    k,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=x[0],
        k=k,
    )


# vsplit
@handle_frontend_test(
    fn_tree="jax.numpy.vsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=2, axis=0, allow_none=False, is_mod_split=True
    ),
    test_with_out=st.just(False),
)
def test_jax_vsplit(
    *,
    dtype_value,
    indices_or_sections,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
    )
