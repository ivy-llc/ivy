# global
from hypothesis import strategies as st, assume
import hypothesis.extra.numpy as nph
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
from ivy.functional.ivy.experimental.manipulation import _check_bounds
from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import _get_splits


# Helpers #
# ------- #


# moveaxis
@handle_test(
    fn_tree="functional.ivy.experimental.moveaxis",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
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
    test_gradients=st.just(False),
)
def test_moveaxis(
    *, dtype_and_a, source, destination, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, a = dtype_and_a
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        a=a[0],
        source=source,
        destination=destination,
    )


# heaviside
@handle_test(
    fn_tree="functional.ivy.experimental.heaviside",
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
    test_gradients=st.just(False),
)
def test_heaviside(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x1=x[0],
        x2=x[0],
    )


# flipud
@handle_test(
    fn_tree="functional.ivy.experimental.flipud",
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    test_gradients=st.just(False),
)
def test_flipud(*, dtype_and_m, test_flags, backend_fw, fn_name, on_device):
    input_dtype, m = dtype_and_m
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        m=m[0],
    )


# vstack
@handle_test(
    fn_tree="functional.ivy.experimental.vstack",
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=helpers.get_shape(
            min_num_dims=1,
        ),
        shared_dtype=True,
        num_arrays=helpers.ints(min_value=2, max_value=10),
    ),
    test_gradients=st.just(False),
)
def test_vstack(*, dtype_and_m, test_flags, backend_fw, fn_name, on_device):
    input_dtype, m = dtype_and_m
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        arrays=m,
    )


# hstack
@handle_test(
    fn_tree="functional.ivy.experimental.hstack",
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shared_dtype=True,
        num_arrays=helpers.ints(min_value=2, max_value=10),
        shape=helpers.get_shape(
            min_num_dims=1,
        ),
    ),
    test_gradients=st.just(False),
)
def test_hstack(dtype_and_m, test_flags, backend_fw, fn_name, on_device):
    input_dtype, m = dtype_and_m
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        arrays=m,
    )


@st.composite
def _get_dtype_values_k_axes_for_rot90(
    draw,
    available_dtypes,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
):
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    k = draw(helpers.ints(min_value=-4, max_value=4))
    axes = draw(
        st.lists(
            helpers.ints(min_value=-len(shape), max_value=len(shape) - 1),
            min_size=2,
            max_size=2,
            unique=True,
        ).filter(lambda axes: abs(axes[0] - axes[1]) != len(shape))
    )
    dtype = draw(st.sampled_from(draw(available_dtypes)))
    values = draw(
        helpers.array_values(
            dtype=dtype,
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            allow_inf=allow_inf,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
            large_abs_safety_factor=72,
            small_abs_safety_factor=72,
            safety_factor_scale="log",
        )
    )
    return [dtype], values, k, axes


# rot90
@handle_test(
    fn_tree="functional.ivy.experimental.rot90",
    dtype_m_k_axes=_get_dtype_values_k_axes_for_rot90(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    test_gradients=st.just(False),
)
def test_rot90(dtype_m_k_axes, test_flags, backend_fw, fn_name, on_device):
    input_dtype, m, k, axes = dtype_m_k_axes
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        m=m,
        k=k,
        axes=tuple(axes),
    )


# top_k
@handle_test(
    fn_tree="functional.ivy.experimental.top_k",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    k=helpers.ints(min_value=1, max_value=4),
    largest=st.booleans(),
    sorted=st.booleans(),
    test_gradients=st.just(False),
)
def test_top_k(
    *, dtype_x_axis, k, largest, sorted, test_flags, backend_fw, fn_name, on_device
):
    dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        k=k,
        axis=axis,
        largest=largest,
        sorted=sorted,
    )


# fliplr
@handle_test(
    fn_tree="functional.ivy.experimental.fliplr",
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
    ),
    test_gradients=st.just(False),
)
def test_fliplr(*, dtype_and_m, test_flags, backend_fw, fn_name, on_device):
    input_dtype, m = dtype_and_m
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        m=m[0],
    )


# i0
@handle_test(
    fn_tree="functional.ivy.experimental.i0",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    test_gradients=st.just(False),
)
def test_i0(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


@st.composite
def _flatten_data_helper(draw):
    mixed_fn_compos = draw(st.booleans())
    is_torch_backend = ivy.current_backend_str() == "torch"

    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "valid", mixed_fn_compos=mixed_fn_compos
            ),
            shape=st.shared(helpers.get_shape(), key="flatten_shape"),
        )
    )
    axes = draw(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(), key="flatten_shape"),
            min_size=2,
            max_size=2,
            unique=False,
            force_tuple=True,
        )
    )
    order = draw(st.sampled_from(["C", "F"]))
    if not mixed_fn_compos and is_torch_backend:
        order = "C"
    return dtype_and_x, axes, order


@handle_test(
    fn_tree="functional.ivy.experimental.flatten",
    data=_flatten_data_helper(),
)
def test_flatten(
    *,
    data,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    (input_dtypes, x), axes, order = data
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        start_dim=axes[0],
        end_dim=axes[1],
        order=order,
    )


def st_tuples(elements, *, min_size=0, max_size=None, unique_by=None, unique=False):
    return st.lists(
        elements,
        min_size=min_size,
        max_size=max_size,
        unique_by=unique_by,
        unique=unique,
    ).map(tuple)


def _st_tuples_or_int(n_pairs, min_val=0):
    return st.one_of(
        st_tuples(
            st.tuples(
                st.integers(min_value=min_val, max_value=4),
                st.integers(min_value=min_val, max_value=4),
            ),
            min_size=n_pairs,
            max_size=n_pairs,
        ),
        helpers.ints(min_value=min_val, max_value=4),
    )


@st.composite
def _pad_helper(draw):
    mode = draw(
        st.sampled_from(
            [
                "constant",
                "dilated",
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
    if mode in ["median", "minimum", "maximum", "linear_ramp"]:
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
        ).filter(lambda x: x[0][0] not in ["float16", "bfloat16"])
    )
    ndim = len(shape)
    min_dim = min(shape)
    if mode == "dilated":
        pad_width = draw(
            st.lists(
                st.tuples(
                    st.integers(min_value=-min_dim, max_value=min_dim),
                    st.integers(min_value=-min_dim, max_value=min_dim),
                    st.integers(min_value=0, max_value=min_dim),
                ),
                min_size=ndim,
                max_size=ndim,
            )
        )
        constant_values = draw(
            helpers.number(
                min_value=0,
                max_value=100,
            ).filter(lambda _x: ivy.as_ivy_dtype(type(_x)) == dtype[0])
        )
    else:
        pad_width = draw(_st_tuples_or_int(ndim))
        constant_values = draw(_st_tuples_or_int(ndim))
    stat_length = draw(_st_tuples_or_int(ndim, min_val=2))
    end_values = draw(_st_tuples_or_int(ndim))
    return dtype, input[0], pad_width, stat_length, constant_values, end_values, mode


@handle_test(
    fn_tree="functional.ivy.experimental.pad",
    ground_truth_backend="numpy",
    dtype_and_input_and_other=_pad_helper(),
    reflect_type=st.sampled_from(["even", "odd"]),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_pad(
    *,
    dtype_and_input_and_other,
    reflect_type,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    (
        dtype,
        input,
        pad_width,
        stat_length,
        constant_values,
        end_values,
        mode,
    ) = dtype_and_input_and_other
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        input=input,
        pad_width=pad_width,
        mode=mode,
        stat_length=stat_length,
        constant_values=constant_values,
        end_values=end_values,
        reflect_type=reflect_type,
    )


# vsplit
@handle_test(
    fn_tree="functional.ivy.experimental.vsplit",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(allow_none=False, min_num_dims=2, axis=0),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_vsplit(
    dtype_and_x, indices_or_sections, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        on_device=on_device,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        indices_or_sections=indices_or_sections,
    )


# dsplit
@handle_test(
    fn_tree="functional.ivy.experimental.dsplit",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=3), key="value_shape"),
    ),
    indices_or_sections=_get_splits(allow_none=False, min_num_dims=3, axis=2),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_dsplit(
    dtype_and_x, indices_or_sections, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        indices_or_sections=indices_or_sections,
    )


# atleast_1d
@handle_test(
    fn_tree="functional.ivy.experimental.atleast_1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=5),
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_atleast_1d(dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, arrays = dtype_and_x
    kw = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtypes)):
        kw["x{}".format(i)] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(kw)
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        **kw,
    )


# dstack
@handle_test(
    fn_tree="functional.ivy.experimental.dstack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shared_dtype=True,
        num_arrays=helpers.ints(min_value=1, max_value=10),
        shape=helpers.get_shape(
            min_num_dims=1,
        ),
    ),
    test_gradients=st.just(False),
)
def test_dstack(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        arrays=x,
    )


# atleast_2d
@handle_test(
    fn_tree="functional.ivy.experimental.atleast_2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=5),
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_atleast_2d(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, arrays = dtype_and_x
    kw = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtypes)):
        kw["x{}".format(i)] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(kw)
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        **kw,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.atleast_3d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=5),
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_atleast_3d(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, arrays = dtype_and_x
    arrys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtypes)):
        arrys["x{}".format(i)] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arrys)
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        **arrys,
    )


# take_along_axis
@handle_test(
    fn_tree="functional.ivy.experimental.take_along_axis",
    dtype_x_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
        valid_bounds=False,
    ),
    mode=st.sampled_from(["clip", "fill", "drop"]),
    ground_truth_backend="jax",
    test_gradients=st.just(False),
)
def test_take_along_axis(
    *,
    dtype_x_indices_axis,
    mode,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtypes, x, indices, axis, _ = dtype_x_indices_axis
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        arr=x,
        indices=indices,
        axis=axis,
        mode=mode,
    )


# hsplit
# TODO: there is a failure with paddle (dtype('int32')) caused by the `_get_splits`
#  method which returns a numpy array with a numpy dtype
@handle_test(
    fn_tree="functional.ivy.experimental.hsplit",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(allow_none=False, min_num_dims=2, axis=1),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_hsplit(
    dtype_and_x, indices_or_sections, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    if (
        not isinstance(indices_or_sections, int)
        and not isinstance(indices_or_sections, list)
        and indices_or_sections is not None
    ):
        input_dtype = [*input_dtype, indices_or_sections.dtype]
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        indices_or_sections=indices_or_sections,
    )


# broadcast_shapes
@handle_test(
    fn_tree="functional.ivy.experimental.broadcast_shapes",
    shapes=nph.mutually_broadcastable_shapes(
        num_shapes=4, min_dims=1, max_dims=5, min_side=1, max_side=5
    ),
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_broadcast_shapes(*, shapes, test_flags, backend_fw, fn_name, on_device):
    shape, _ = shapes
    shapes = {f"shape{i}": shape[i] for i in range(len(shape))}
    test_flags.num_positional_args = len(shapes)
    helpers.test_function(
        input_dtypes=["int64"],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        **shapes,
    )


@handle_test(
    fn_tree="expand",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=False),
        shape=st.shared(
            helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=5,
            ),
            key="value_shape",
        ),
    ),
    shape=st.shared(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
        ),
        key="value_shape",
    ),
    container_flags=st.just([False]),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_expand(*, dtype_and_x, shape, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        shape=shape,
    )


@st.composite
def _as_strided_helper(draw):
    dtype, x = draw(helpers.dtype_and_values(min_num_dims=1, max_num_dims=5))
    x = x[0]
    itemsize = x.itemsize
    shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=5))
    new_ndim = len(shape)
    strides = draw(
        st.lists(
            st.integers(min_value=1, max_value=16),
            min_size=new_ndim,
            max_size=new_ndim,
        ).filter(lambda x: all(x[i] % itemsize == 0 for i in range(new_ndim)))
    )
    assume(_check_bounds(x.shape, shape, strides, itemsize))
    return dtype, x, shape, strides


@handle_test(
    fn_tree="as_strided",
    all_args=_as_strided_helper(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    ground_truth_backend="numpy",
)
def test_as_strided(*, all_args, test_flags, backend_fw, fn_name, on_device):
    dtype, x, shape, strides = all_args
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x,
        shape=shape,
        strides=strides,
    )


@st.composite
def _concat_from_sequence_helper(draw):
    dtypes, arrays, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=helpers.ints(min_value=1, max_value=6),
            ret_shape=True,
            min_num_dims=2,
            min_dim_size=2,
            shared_dtype=True,
        )
    )
    axis = draw(
        helpers.get_axis(
            shape=shape,
            force_int=True,
        )
    )
    return dtypes, arrays, axis


# concat_from_sequence
@handle_test(
    fn_tree="functional.ivy.experimental.concat_from_sequence",
    dtypes_arrays_axis=_concat_from_sequence_helper(),
    new_axis=st.integers(min_value=0, max_value=1),
    container_flags=st.just([False]),
    test_instance_method=st.just(False),
)
def test_concat_from_sequence(
    *, dtypes_arrays_axis, new_axis, test_flags, backend_fw, fn_name, on_device
):
    dtypes, arrays, axis = dtypes_arrays_axis

    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        input_sequence=arrays,
        new_axis=new_axis,
        axis=axis,
    )


@st.composite
def _associative_scan_helper(draw):
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("float"))),
            key="shared_dtype",
        ).filter(lambda _x: "float16" not in _x)
    )
    random_size = draw(
        st.shared(helpers.ints(min_value=1, max_value=5), key="shared_size")
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=1, max_value=5), key="shared_size")
    )
    shape = tuple([random_size, shared_size, shared_size])
    matrix = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=shape,
            min_value=1,
            max_value=10,
        )
    )
    axis = draw(
        helpers.get_axis(
            shape=shape,
            allow_neg=False,
            force_int=True,
        ).filter(lambda _x: _x < len(shape) - 2)
    )
    return [input_dtype], matrix, axis


# associative_scan
@handle_test(
    fn_tree="functional.ivy.experimental.associative_scan",
    dtype_elems_axis=_associative_scan_helper(),
    fn=st.sampled_from([ivy.matmul, ivy.multiply, ivy.add]),
    reverse=st.booleans(),
    test_with_out=st.just(False),
    ground_truth_backend="jax",
)
def test_associative_scan(
    *, dtype_elems_axis, fn, reverse, fn_name, test_flags, backend_fw, on_device
):
    dtype, elems, axis = dtype_elems_axis
    helpers.test_function(
        fn_name=fn_name,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        input_dtypes=dtype,
        elems=elems,
        fn=fn,
        reverse=reverse,
        axis=axis,
    )


# unique_consecutive
@handle_test(
    fn_tree="unique_consecutive",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=2,
        force_int_axis=True,
        valid_axis=True,
    ),
    none_axis=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    ground_truth_backend="torch",
)
def test_unique_consecutive(
    *, dtype_x_axis, none_axis, test_flags, backend_fw, fn_name, on_device
):
    dtype, x, axis = dtype_x_axis
    if none_axis:
        axis = None
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        axis=axis,
    )


# fill_diag
@handle_test(
    fn_tree="fill_diagonal",
    dt_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=4,
        min_dim_size=3,
        max_dim_size=3,
    ),
    v=st.sampled_from([1, 2, 3, 10]),
    wrap=st.booleans(),
    test_with_out=st.just(False),
)
def test_fill_diagonal(
    *,
    dt_a,
    v,
    wrap,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dt, a = dt_a
    helpers.test_function(
        input_dtypes=dt,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        a=a[0],
        v=v,
        wrap=wrap,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.unfold",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        allow_neg_axes=False,
        force_int_axis=True,
    ),
)
def test_unfold(*, dtype_values_axis, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input, axis = dtype_values_axis
    if axis is None:
        axis = 0
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtype,
        x=input[0],
        mode=axis,
    )


@st.composite
def _fold_data(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=3
        )
    )
    mode = draw(helpers.ints(min_value=0, max_value=len(shape) - 1))
    reduced_dims = int(ivy.prod(shape[0:mode]) * ivy.prod(shape[mode + 1 :]))
    unfolded_shape = (shape[mode], reduced_dims)
    dtype, input = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"), shape=unfolded_shape
        )
    )
    return dtype, input, shape, mode


@handle_test(
    fn_tree="functional.ivy.experimental.fold",
    data=_fold_data(),
)
def test_fold(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input, shape, mode = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtype,
        x=input[0],
        mode=mode,
        shape=shape,
    )


@st.composite
def _partial_unfold_data(draw):
    dtype, input = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=1,
        )
    )
    input = input[0]
    ndims = len(input.shape)
    mode_and_skip_begin = draw(
        st.lists(
            helpers.ints(min_value=0, max_value=ndims - 1), min_size=2, max_size=2
        ).filter(lambda nums: np.sum(nums) <= ndims - 1)
    )
    skip_begin, mode = sorted(mode_and_skip_begin)
    skip_end = draw(
        helpers.ints(min_value=0, max_value=ndims - (skip_begin + mode) - 1)
    )
    ravel_tensors = draw(st.booleans())
    return dtype, input, mode, skip_begin, skip_end, ravel_tensors


@handle_test(
    fn_tree="functional.ivy.experimental.partial_unfold",
    data=_partial_unfold_data(),
)
def test_partial_unfold(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input, axis, skip_begin, skip_end, ravel_tensors = data
    test_flags.instance_method = False
    helpers.test_function(
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtype,
        input=input,
        mode=axis,
        skip_begin=skip_begin,
        skip_end=skip_end,
        ravel_tensors=ravel_tensors,
    )


@st.composite
def _partial_fold_data(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=3
        )
    )
    ndims = len(shape)
    mode_and_skip_begin = draw(
        st.lists(
            helpers.ints(min_value=0, max_value=ndims - 1), min_size=2, max_size=2
        ).filter(lambda nums: np.sum(nums) <= ndims - 1)
    )
    skip_begin, mode = sorted(mode_and_skip_begin)
    skip_end = draw(
        helpers.ints(min_value=0, max_value=ndims - (skip_begin + mode) - 1)
    )
    if skip_end != 0:
        reduced_dims = int(
            ivy.prod(shape[skip_begin : skip_begin + mode])
            * ivy.prod(shape[skip_begin + mode + 1 : -skip_end])
        )
        unfolded_shape = (
            *shape[:skip_begin],
            shape[skip_begin + mode],
            reduced_dims,
            *shape[-skip_end:],
        )
    else:
        reduced_dims = int(
            ivy.prod(shape[skip_begin : skip_begin + mode])
            * ivy.prod(shape[skip_begin + mode + 1 :])
        )
        unfolded_shape = (*shape[:skip_begin], shape[skip_begin + mode], reduced_dims)

    dtype, input = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"), shape=unfolded_shape
        )
    )
    return dtype, input, skip_begin, shape, mode


@handle_test(
    fn_tree="functional.ivy.experimental.partial_fold",
    data=_partial_fold_data(),
)
def test_partial_fold(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input, skip_begin, shape, mode = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtype,
        input=input,
        mode=mode,
        shape=shape,
        skip_begin=skip_begin,
    )


@st.composite
def _partial_tensor_to_vec_data(draw):
    input_dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"), min_num_dims=1, ret_shape=True
        )
    )
    ndims = len(shape)
    skip_begin = draw(helpers.ints(min_value=0, max_value=ndims - 1))
    skip_end = draw(helpers.ints(min_value=0, max_value=ndims - 1 - skip_begin))
    return input_dtype, input, skip_begin, skip_end


# TODO Add container and instance methods
@handle_test(
    fn_tree="functional.ivy.experimental.partial_tensor_to_vec",
    data=_partial_tensor_to_vec_data(),
)
def test_partial_tensor_to_vec(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input, skip_begin, skip_end = data
    test_flags.instance_method = False
    helpers.test_function(
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtype,
        input=input,
        skip_begin=skip_begin,
        skip_end=skip_end,
    )


@st.composite
def _partial_vec_to_tensor(draw):
    shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=5))
    numel = int(ivy.prod(shape))
    input_dtype, input = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"), shape=(numel,)
        )
    )
    ndims = len(shape)
    skip_begin = draw(helpers.ints(min_value=0, max_value=ndims - 1))
    return input_dtype, input, shape, skip_begin


# TODO Add container and instance methods
@handle_test(
    fn_tree="functional.ivy.experimental.partial_vec_to_tensor",
    data=_partial_vec_to_tensor(),
)
def test_partial_vec_to_tensor(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input, shape, skip_begin = data
    test_flags.instance_method = False
    helpers.test_function(
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtype,
        input=input,
        shape=shape,
        skip_begin=skip_begin,
    )


@st.composite
def _matricize_data(draw):
    input_dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"), min_num_dims=1, ret_shape=True
        )
    )
    ndims = len(shape)
    dims = set([*range(ndims)])
    row_modes = set(draw(st.lists(helpers.ints(min_value=0, max_value=ndims - 1))))
    col_modes = dims - row_modes
    return input_dtype, input[0], row_modes, col_modes


# TODO Add container and instance methods
@handle_test(
    fn_tree="functional.ivy.experimental.matricize",
    data=_matricize_data(),
)
def test_matricize(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input, row_modes, column_modes = data
    test_flags.instance_method = False
    helpers.test_function(
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtype,
        input=input,
        row_modes=row_modes,
        column_modes=column_modes,
    )


@st.composite
def _soft_thresholding_data(draw):
    x_min, x_max = 1e-2, 1e2
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            ret_shape=True,
            min_value=x_min,
            max_value=x_max,
        )
    )
    threshold_choice_1 = draw(helpers.floats(min_value=x_min, max_value=x_max))
    t_dtype, threshold_choice_2 = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape,
            min_value=x_min,
            max_value=x_max,
        )
    )
    threshold = draw(st.sampled_from([threshold_choice_1, threshold_choice_2]))
    return x_dtype + t_dtype, x[0], threshold


# TODO Add container and instance methods
@handle_test(
    fn_tree="functional.ivy.experimental.soft_thresholding",
    data=_soft_thresholding_data(),
)
def test_soft_thresholding(*, data, test_flags, backend_fw, fn_name, on_device):
    x_dtype, x, threshold = data
    test_flags.instance_method = False
    helpers.test_function(
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=x_dtype,
        x=x,
        threshold=threshold,
    )
