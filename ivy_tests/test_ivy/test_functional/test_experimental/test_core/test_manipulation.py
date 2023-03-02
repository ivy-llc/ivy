# global
from hypothesis import strategies as st
import hypothesis.extra.numpy as nph

# local
import numpy as np
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


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
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_dim_size=4,
        max_dim_size=10,
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    k=helpers.ints(min_value=1, max_value=4),
    largest=st.booleans(),
    test_gradients=st.just(False),
)
def test_top_k(
    *,
    dtype_and_x,
    axis,
    k,
    largest,
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
        k=k,
        axis=axis,
        largest=largest,
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
        ).filter(lambda x: x[0][0] not in ["float16", "bfloat16"])
    )
    ndim = len(shape)
    pad_width = draw(_st_tuples_or_int(ndim))
    stat_length = draw(_st_tuples_or_int(ndim, min_val=2))
    constant_values = draw(_st_tuples_or_int(ndim))
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


@st.composite
def _get_split_locations(draw, min_num_dims, axis=None):
    """
    Generate valid splits, either by generating an integer that evenly divides the axis
    or a list of split locations.
    """
    shape = draw(
        st.shared(helpers.get_shape(min_num_dims=min_num_dims), key="value_shape")
    )
    if len(shape) == 1:
        axis = draw(st.just(0))
    elif ivy.exists(axis):
        axis = draw(st.just(axis))
    else:
        axis = draw(
            st.shared(helpers.get_axis(shape=shape, force_int=True), key="target_axis")
        )

    @st.composite
    def get_int_split(draw):
        if shape[axis] == 0:
            return 0
        factors = []
        for i in range(1, shape[axis] + 1):
            if shape[axis] % i == 0:
                factors.append(i)
        return draw(st.sampled_from(factors))

    @st.composite
    def get_list_split(draw):
        return draw(
            st.lists(
                st.integers(min_value=0, max_value=shape[axis]),
                min_size=0,
                max_size=shape[axis],
                unique=True,
            ).map(sorted)
        )

    return draw(get_list_split() | get_int_split())


# vsplit
@handle_test(
    fn_tree="functional.ivy.experimental.vsplit",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    indices_or_sections=helpers.get_shape(
        min_num_dims=1, max_num_dims=3, min_dim_size=1, max_dim_size=3
    ),
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
    indices_or_sections = sorted(indices_or_sections)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
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
    indices_or_sections=_get_split_locations(min_num_dims=3, axis=2),
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
    ),
    test_gradients=st.just(False),
)
def test_take_along_axis(
    *,
    dtype_x_indices_axis,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtypes, x, indices, axis, _ = dtype_x_indices_axis
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtypes,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        arr=x,
        indices=indices,
        axis=axis,
    )


# hsplit
@handle_test(
    fn_tree="functional.ivy.experimental.hsplit",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    indices_or_sections=helpers.get_shape(
        min_num_dims=1, max_num_dims=3, min_dim_size=1, max_dim_size=3
    ),
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
    indices_or_sections = sorted(indices_or_sections)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        indices_or_sections=indices_or_sections,
    )


# dstack
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
    shapes, _ = shapes
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=["int64"],
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        shapes=shapes,
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
