# global
from hypothesis import strategies as st, assume
import hypothesis.extra.numpy as nph
import numpy as np
from typing import Sequence

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
    *,
    dtype_and_a,
    source,
    destination,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, a = dtype_and_a
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        a=a[0],
        source=source,
        destination=destination,
    )


# ndenumerate
@handle_test(
    fn_tree="functional.ivy.experimental.ndenumerate",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
    ),
)
def test_ndenumerate(dtype_and_x):
    values = dtype_and_x[1][0]
    for (index1, x1), (index2, x2) in zip(
        np.ndenumerate(values), ivy.ndenumerate(values)
    ):
        assert index1 == index2 and x1 == x2


# ndindex
@handle_test(
    fn_tree="functional.ivy.experimental.ndindex",
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        ret_shape=True,
    ),
)
def test_ndindex(dtype_x_shape):
    shape = dtype_x_shape[2]
    for index1, index2 in zip(np.ndindex(shape), ivy.ndindex(shape)):
        assert index1 == index2


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
def test_heaviside(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
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
def test_flipud(
    *,
    dtype_and_m,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, m = dtype_and_m
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
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
def test_vstack(
    *,
    dtype_and_m,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, m = dtype_and_m
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
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
def test_hstack(
    dtype_and_m,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, m = dtype_and_m
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
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
def test_rot90(
    dtype_m_k_axes,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, m, k, axes = dtype_m_k_axes
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
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
    *,
    dtype_x_axis,
    k,
    largest,
    sorted,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
def test_fliplr(
    *,
    dtype_and_m,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, m = dtype_and_m
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
def test_i0(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# flatten
@handle_test(
    fn_tree="functional.ivy.experimental.flatten",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="flatten_shape"),
    ),
    axes=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="flatten_shape"),
        min_size=2,
        max_size=2,
        unique=False,
        force_tuple=True,
    ),
    order=st.sampled_from(["C", "F"]),
    test_gradients=st.just(False),
    number_positional_args=st.just(1),
)
def test_flatten(
    *,
    dtype_and_x,
    axes,
    order,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtypes, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        fw=backend_fw,
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
    ground_truth_backend,
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
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
    dtype_and_x,
    indices_or_sections,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    if isinstance(indices_or_sections, Sequence):
        indices_or_sections = sorted(indices_or_sections)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        on_device=on_device,
        test_flags=test_flags,
        fw=backend_fw,
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
    dtype_and_x,
    indices_or_sections,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    if isinstance(indices_or_sections, Sequence):
        indices_or_sections = sorted(indices_or_sections)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
def test_atleast_1d(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtypes, arrays = dtype_and_x
    kw = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtypes)):
        kw["x{}".format(i)] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(kw)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        fw=backend_fw,
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
def test_dstack(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
def test_atleast_2d(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtypes, arrays = dtype_and_x
    kw = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtypes)):
        kw["x{}".format(i)] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(kw)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        **kw,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.atleast_3d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=5),
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_atleast_3d(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtypes, arrays = dtype_and_x
    kw = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtypes)):
        kw["x{}".format(i)] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(kw)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        **kw,
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
        ground_truth_backend="jax",
        input_dtypes=dtypes,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        arr=x,
        indices=indices,
        axis=axis,
        mode=mode,
    )


# hsplit
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
    dtype_and_x,
    indices_or_sections,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    if isinstance(indices_or_sections, Sequence):
        indices_or_sections = sorted(indices_or_sections)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
def test_broadcast_shapes(
    *,
    shapes,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    shape, _ = shapes
    shapes = {f"shape{i}": shape[i] for i in range(len(shape))}
    test_flags.num_positional_args = len(shapes)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=["int64"],
        test_flags=test_flags,
        fw=backend_fw,
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
def test_expand(
    *,
    dtype_and_x,
    shape,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
def test_as_strided(
    *,
    all_args,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, shape, strides = all_args
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
    *,
    dtypes_arrays_axis,
    new_axis,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtypes, arrays, axis = dtypes_arrays_axis

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtypes,
        test_flags=test_flags,
        fw=backend_fw,
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
    *,
    dtype_elems_axis,
    fn,
    reverse,
    fn_name,
    test_flags,
    backend_fw,
    on_device,
    ground_truth_backend,
):
    dtype, elems, axis = dtype_elems_axis
    helpers.test_function(
        fn_name=fn_name,
        test_flags=test_flags,
        fw=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
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
    *,
    dtype_x_axis,
    none_axis,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, axis = dtype_x_axis
    if none_axis:
        axis = None
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        axis=axis,
    )
