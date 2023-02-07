# global
import sys
import ivy
from hypothesis import assume, strategies as st
from ivy.functional.frontends.tensorflow.nn import _convolution_broadcast_helper
from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_nn import _x_and_filters
import numpy as np
import math

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Acos
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Acos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Acos(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Acosh
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Acosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Acosh(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Add
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Add(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# for data generation
dtype_shared = st.shared(st.sampled_from(helpers.get_dtypes("numeric")), key="dtype")


@st.composite
def _get_shared_dtype(draw):
    return st.shared(st.sampled_from(draw(helpers.get_dtypes("numeric"))), key="dtype")


# BroadcastTo
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.BroadcastTo",
    array_and_shape=helpers.array_and_broadcastable_shape(_get_shared_dtype()),
    test_with_out=st.just(False),
)
def test_tensorflow_BroadcastTo(  # NOQA
    *,
    array_and_shape,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    x, to_shape = array_and_shape
    helpers.test_frontend_function(
        input_dtypes=[x.dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        shape=to_shape,
    )


# noinspection DuplicatedCode
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
        helpers.array_dtypes(
            available_dtypes=draw(helpers.get_dtypes("float")), shared_dtype=True
        )
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


# Concat
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Concat",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    test_with_out=st.just(False),
)
def test_tensorflow_Concat(  # NOQA
    *,
    xs_n_input_dtypes_n_unique_idx,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        concat_dim=unique_idx,
        values=xs,
    )


# Cos
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Cos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Cos(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Rsqrt
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Rsqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Rsqrt(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Cosh
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Cosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Cosh(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(draw(helpers.get_dtypes("numeric"))), length=1
            ),
            key="dtype",
        )
    )


# Div
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Div",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Div(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@st.composite
def _fill_value(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_uint_dtype(dtype):
        return draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype):
        return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


# fill
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Fill",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        min_dim_size=1,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    test_with_out=st.just(False),
)
def test_tensorflow_Fill(  # NOQA
    *,
    shape,
    fill_value,
    dtypes,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-05,
        dims=shape,
        value=fill_value,
    )


# Asin
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Asin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Asin(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        x=x[0],
    )


# argmax
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.ArgMax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
        allow_inf=False,
    ),
    output_type=st.sampled_from(["int16", "int32", "int64"]),
    test_with_out=st.just(False),
)
def test_tensorflow_ArgMax(  # NOQA
    *,
    dtype_x_axis,
    output_type,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dimension=axis,
        output_type=output_type,
    )


# ArgMin
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.ArgMin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
        allow_inf=False,
    ),
    output_type=st.sampled_from(["int32", "int64"]),
    test_with_out=st.just(False),
)
def test_tensorflow_ArgMin(  # NOQA
    *,
    dtype_x_axis,
    output_type,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dimension=axis,
        output_type=output_type,
    )


# Atan
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Atan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Atan(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        x=x[0],
    )


# BitwiseAnd
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.BitwiseAnd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_BitwiseAnd(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# BitwiseOr
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.BitwiseOr",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_BitwiseOr(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# BitwiseXor
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.BitwiseXor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_BitwiseXor(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# Atanh
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Atanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Atanh(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        x=x[0],
    )


# Tan
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Tan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Tan(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Square
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Square",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Square(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


@st.composite
def _squeeze_helper(draw):
    shape = draw(st.shared(helpers.get_shape(), key="value_shape"))
    valid_axes = []
    for index, axis in enumerate(shape):
        if axis == 1:
            valid_axes.append(index)
    valid_axes.insert(0, None)
    return draw(st.sampled_from(valid_axes))


# Squeeze
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Squeeze",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=_squeeze_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_Squeeze(  # NOQA
    dtype_value,
    axis,
    frontend,
    test_flags,
    fn_tree,
):
    dtype, xs = dtype_value
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=xs[0],
        axis=axis,
    )


# Sign
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Sign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Sign(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


@st.composite
def _get_splits(draw, as_list=False):
    """
    Generate valid splits, either by generating an integer that evenly divides the axis
    or a list of splits that sum to the length of the axis being split.
    """
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"))
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
        num_or_size_splits = []
        while sum(num_or_size_splits) < shape[axis]:
            split_value = draw(
                helpers.ints(
                    min_value=1,
                    max_value=shape[axis] - sum(num_or_size_splits),
                )
            )
            num_or_size_splits.append(split_value)
        return num_or_size_splits

    if as_list:
        return draw(get_list_split())
    else:
        return draw(get_int_split())


# Split
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Split",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    axis=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    num_splits=_get_splits(),
)
def test_tensorflow_Split(  # NOQA
    *,
    dtype_and_x,
    axis,
    num_splits,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, value = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        value=value[0],
        axis=axis,
        num_split=num_splits,
    )


# SplitV
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.SplitV",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    axis=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    size_splits=_get_splits(as_list=True),
    test_with_out=st.just(False),
)
def test_tensorflow_SplitV(  # NOQA
    *,
    dtype_and_x,
    axis,
    size_splits,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, value = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        value=value[0],
        axis=axis,
        size_splits=size_splits,
        num_split=len(size_splits),
    )


# Sqrt
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Sqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Sqrt(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Tanh
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Tanh(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


@st.composite
def _permute_dims_helper(draw):
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="shape"))
    dims = [x for x in range(len(shape))]
    permutation = draw(st.permutations(dims))
    return permutation


# Transpose
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Transpose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    perm=_permute_dims_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_transpose(  # NOQA
    *,
    dtype_and_x,
    perm,
    frontend,
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
        x=x[0],
        perm=perm,
    )


# Maximum
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Maximum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Maximum(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# Minimum
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Minimum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Minimum(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# Sub
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Sub",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2, shared_dtype=True
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Sub(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# Less
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Less",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Less(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# LessEqual
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.LessEqual",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_LessEqual(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# Floor
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Floor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Floor(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# FloorDiv
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.FloorDiv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_FloorDiv(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# Exp
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Exp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Exp(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Expm1
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Expm1",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Expm1(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Log
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Log(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Log1p
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Log1p",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Log1p(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Sinh
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Sinh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_Sinh(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# RealDiv
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.RealDiv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_RealDiv(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# Reshape
@st.composite
def _reshape_helper(draw):
    # generate a shape s.t len(shape) > 0
    shape = draw(helpers.get_shape(min_num_dims=1))
    reshape_shape = draw(helpers.reshape_shapes(shape=shape))
    dtype = draw(helpers.array_dtypes(num_arrays=1))
    x = draw(helpers.array_values(dtype=dtype[0], shape=shape))
    return x, dtype, reshape_shape


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Reshape",
    test_with_out=st.just(False),
    x_reshape=_reshape_helper(),
)
def test_tensorflow_Reshape(  # NOQA
    *,
    x_reshape,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    x, dtype, shape = x_reshape
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x,
        shape=shape,
    )


# ZerosLike
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.ZerosLike",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_zeros_like(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        x=x[0],
    )


# LogicalOr
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.LogicalOr",
    dtype_and_x=helpers.dtype_and_values(
        dtype=["bool", "bool"],
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_LogicalOr(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# LogicalNot
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.LogicalNot",
    dtype_and_x=helpers.dtype_and_values(
        dtype=["bool"],
        num_arrays=1,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_LogicalNot(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Shape
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Shape",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Shape(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.ShapeN",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), max_num_dims=4
    ),
    output_dtype=st.sampled_from(["int32", "int64"]),
    test_with_out=st.just(False),
)
def test_tensorflow_ShapeN(  # NOQA
    *,
    dtype_and_x,
    output_dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        out_type=output_dtype,
    )


# AddN
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.AddN",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_value=-1e04,
        max_value=1e04,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_AddN(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        inputs=x[0],
    )


# Neg
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Neg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=[
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
        ],
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Neg(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


# Equal
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Equal(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# NotEqual
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.NotEqual",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_NotEqual(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# Cumsum
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    exclusive=st.booleans(),
    reverse=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_Cumsum(  # NOQA
    *,
    dtype_x_axis,
    exclusive,
    reverse,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-02,
        x=x[0],
        axis=axis,
        exclusive=exclusive,
        reverse=reverse,
    )


# Relu
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Relu(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        features=x[0],
    )


# MatMul
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.MatMul",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=[
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        ],
        shape=(3, 3),
        num_arrays=2,
        shared_dtype=True,
    ),
    transpose_a=st.booleans(),
    transpose_b=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_MatMul(  # NOQA
    *,
    dtype_and_x,
    transpose_a,
    transpose_b,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-2,
        a=x[0],
        b=x[1],
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


# Cumprod
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Cumprod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    exclusive=st.booleans(),
    reverse=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_Cumprod(  # NOQA
    *,
    dtype_x_axis,
    exclusive,
    reverse,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        exclusive=exclusive,
        reverse=reverse,
    )


# Gather
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Gather",
    params_indices_others=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        disable_random_axis=True,
        axis_zero=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Gather(  # NOQA
    *,
    params_indices_others,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtypes, params, indices = params_indices_others
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        params=params,
        indices=indices,
        validate_indices=True,
    )


# Greater
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Greater",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Greater(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# GreaterEqual
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.GreaterEqual",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_GreaterEqual(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# Mean
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Mean",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-10,
        max_value=3,
    ),
    keep_dims=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_Mean(  # NOQA
    *,
    dtype_x_axis,
    keep_dims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        axis=axis,
        keep_dims=keep_dims,
        rtol=1e-02,
        atol=1e-02,
    )


# Identity
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Identity",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Identity(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        input=x[0],
    )


# IdentityN
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.IdentityN",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_IdentityN(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        input=x,
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Inv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True)
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Inv(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        x=x[0],
    )


# reciprocal
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Reciprocal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_value=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Reciprocal(  # NOQA
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.OnesLike",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True)
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_OnesLike(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Cholesky",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Cholesky(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    x = x[0]
    x = (
        np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    )  # make symmetric positive-definite

    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        rtol=1e-4,
        atol=1e-4,
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Mul",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Mul(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Min",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    keep_dims=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_Min(  # NOQA
    *,
    dtype_x_axis,
    keep_dims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        axis=axis,
        keep_dims=keep_dims,
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Max",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    keep_dims=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_Max(  # NOQA
    *,
    dtype_x_axis,
    keep_dims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        axis=axis,
        keep_dims=keep_dims,
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.LeftShift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_LeftShift(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.MatrixDeterminant",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
        min_value=-5,
        max_value=5,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_MatrixDeterminant(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        input=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.NthElement",
    array_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric", full=True),
        indices_dtypes=["int32"],
        min_num_dims=1,
        min_dim_size=1,
        disable_random_axis=True,
    ),
    reverse=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_NthElement(  # NOQA
    *,
    array_indices_axis,
    reverse,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, n = array_indices_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        n=n.flatten()[0],
        reverse=reverse,
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Invert",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer", full=True),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Invert(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.InvGrad",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=1,
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_InvGrad(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        y=x[0],
        dy=x[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Ceil",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Ceil(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Diag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=[
            "float32",
            "float64",
            "int32",
            "int64",
        ],
        min_num_dims=1,
        max_num_dims=1,
        min_value=-1e30,
        max_value=1e30,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Diag(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        diagonal=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.RightShift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
        min_value=0,
        max_value=8,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_RightShift(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@st.composite
def _pow_helper_shared_dtype(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float", full=True),
            num_arrays=2,
            shared_dtype=True,
        )
    )
    dtype1, dtype2 = dtype
    x1, x2 = x
    if "int" in dtype2:
        x2 = ivy.nested_map(x2, lambda x: abs(x), include_derived={list: True})

    if ivy.is_int_dtype(dtype2):
        max_val = ivy.iinfo(dtype2).max
    else:
        max_val = ivy.finfo(dtype2).max
    max_x1 = np.max(np.abs(x1))
    if max_x1 in [0, 1]:
        max_value = None
    else:
        max_value = int(math.log(max_val) / math.log(max_x1))
        if abs(max_value) > abs(max_val) / 40 or max_value < 0:
            max_value = None

    return [dtype1, dtype2], [x1, x2]


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Pow",
    dtype_and_x=_pow_helper_shared_dtype(),
    test_with_out=st.just(False),
)
def test_tensorflow_Pow(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Sum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    keep_dims=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_Sum(  # NOQA
    *,
    dtype_x_axis,
    keep_dims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        axis=axis,
        keep_dims=keep_dims,
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.TruncateDiv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"), num_arrays=2, shared_dtype=True
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_TruncateDiv(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, xs = dtype_and_x
    # prevent too close to zero
    assume(not np.any(np.isclose(xs[1], 0)))

    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.MatrixInverse",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=helpers.ints(min_value=2, max_value=10).map(lambda x: tuple([x, x])),
    ).filter(lambda x: np.linalg.cond(x[1][0].tolist()) < 1 / sys.float_info.epsilon),
    adjoint=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_MatrixInverse(  # NOQA
    *,
    dtype_x,
    adjoint,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        adjoint=adjoint,
        rtol=1e-05,
        atol=1e-04,
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Relu6",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Relu6(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        features=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Round",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Round(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Unpack",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Unpack(  # NOQA
    *,
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        value=x[0],
        num=x[0].shape[axis],
        axis=axis,
    )


# Sigmoid
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Sigmoid(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Softplus",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Softplus(  # NOQA
    *,
    dtype_and_x,
    frontend,
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
        features=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Xdivy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Xdivy(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Xlog1py",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Xlog1py(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Xlogy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Xlogy(  # NOQA
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Pack",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Pack(  # NOQA
    dtype_x_axis,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        values=x,
        axis=axis,
    )


@st.composite
def _pad_helper(draw, return_constant_values=False):
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            min_num_dims=1,
            ret_shape=True,
        )
    )
    ndim = len(shape)
    padding_dtype, paddings = draw(
        helpers.dtype_and_values(
            available_dtypes=["int32", "int64"],
            shape=(ndim, 2),
            min_value=0,
            max_value=10,
        )
    )

    if return_constant_values:
        _, constant_values = draw(
            helpers.dtype_and_values(
                dtype=dtype,
                shape=(1,),
            )
        )
        return dtype, input[0], padding_dtype, paddings[0], constant_values[0][0]

    return dtype, input[0], padding_dtype, paddings[0]


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Pad",
    dtype_x_paddings=_pad_helper(),
    number_positional_args=st.just(0),
    test_with_out=st.just(False),
)
def test_tensorflow_Pad(  # NOQA
    dtype_x_paddings,
    frontend,
    test_flags,
    fn_tree,
):
    dtype, x, padding_dtype, paddings = dtype_x_paddings
    helpers.test_frontend_function(
        input_dtypes=dtype + padding_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=x,
        paddings=paddings,
    )


# EuclideanNorm
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.EuclideanNorm",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=4,
        min_axis=-3,
        max_axis=2,
        valid_axis=True,
        allow_neg_axes=True,
    ),
    keep_dims=st.booleans(),
    test_with_out=st.just(False),
    number_positional_args=st.just(0),
)
def test_tensorflow_EuclideanNorm(
    dtype_values_axis,
    keep_dims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, values, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=values[0],
        axis=axis,
        keep_dims=keep_dims,
    )


# ConcatV2
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.ConcatV2",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    test_with_out=st.just(False),
    number_positional_args=st.just(0),
)
def test_tensorflow_ConcatV2(
    xs_n_input_dtypes_n_unique_idx,
    test_flags,
    frontend,
    fn_tree,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        values=xs,
        axis=unique_idx,
    )


# Conv3D
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Conv3D",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NDHWC"]),
        padding=st.sampled_from(["SAME", "VALID"]),
        type="3d",
        # Tensorflow backprop doesn't support dilations more than 1 on CPU
        dilation_min=1,
        dilation_max=1,
    ),
    test_with_out=st.just(False),
    number_positional_args=st.just(0),
)
def test_tensorflow_Conv3D(
    *,
    x_f_d_df,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df

    # Broadcast stirdes and dilations to correct dims for the ground truth
    # backend func to run correctly
    stride = _convolution_broadcast_helper(
        stride, num_spatial_dims=3, channel_index=4, name="strides"
    )
    dilation = _convolution_broadcast_helper(
        dilation, num_spatial_dims=3, channel_index=4, name="dilations"
    )

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filter=filters,
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilations=dilation,
    )


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Softmax",
    dtype_values_axis=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=2,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_Softmax(
    dtype_values_axis,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, values = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        logits=values[0],
    )


# TODO: Fails with torch backend
# ivy.exceptions.IvyBackendException: torch: constant_pad: constant_pad_nd(): argument
# 'value' (position 3) must be Number, not bfloat16
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.PadV2",
    dtype_x_paddings=_pad_helper(return_constant_values=True),
    test_with_out=st.just(False),
)
def test_tensorflow_PadV2(
    dtype_x_paddings,
    frontend,
    test_flags,
    fn_tree,
):
    dtype, x, padding_dtype, paddings, constant_values = dtype_x_paddings
    helpers.test_frontend_function(
        input_dtypes=dtype + padding_dtype + dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        input=x,
        paddings=paddings,
        constant_values=constant_values,
    )


# Elu
@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.Elu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-3,
        max_value=3,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    name=st.just(None),
    test_with_out=st.just(False),
    number_positional_args=st.just(0),
)
def test_tensorflow_Elu(
    *,
    dtype_and_x,
    name,
    frontend,
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
        features=x[0],
        name=name,
    )
