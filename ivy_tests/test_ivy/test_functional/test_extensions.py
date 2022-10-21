# global
from hypothesis import given, assume, strategies as st

# local
import numpy as np
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Helpers #
# ------- #


@st.composite
def _sparse_coo_indices_values_shape(draw):
    num_elem = draw(helpers.ints(min_value=2, max_value=8))
    dim1 = draw(helpers.ints(min_value=2, max_value=5))
    dim2 = draw(helpers.ints(min_value=5, max_value=10))
    value_dtype = draw(helpers.get_dtypes("numeric", full=False))[0]
    coo_indices = draw(
        helpers.array_values(
            dtype="int64",
            shape=(2, num_elem),
            min_value=0,
            max_value=dim1,
        )
    )
    values = draw(helpers.array_values(dtype=value_dtype, shape=(num_elem,)))
    shape = (dim1, dim2)
    return coo_indices, value_dtype, values, shape


@st.composite
def _sparse_csr_indices_values_shape(draw):
    num_elem = draw(helpers.ints(min_value=2, max_value=8))
    dim1 = draw(helpers.ints(min_value=2, max_value=5))
    dim2 = draw(helpers.ints(min_value=5, max_value=10))
    value_dtype = draw(helpers.get_dtypes("numeric", full=False))[0]
    values = draw(helpers.array_values(dtype=value_dtype, shape=(num_elem,)))
    col_indices = draw(
        helpers.array_values(
            dtype="int64",
            shape=(num_elem,),
            min_value=0,
            max_value=dim2,
        )
    )
    indices = draw(
        helpers.array_values(
            dtype="int64",
            shape=(dim1 - 1,),
            min_value=0,
            max_value=num_elem,
        )
    )
    crow_indices = [0] + sorted(indices) + [num_elem]
    shape = (dim1, dim2)
    return crow_indices, col_indices, value_dtype, values, shape


# coo - to_dense_array
@handle_cmd_line_args
@given(sparse_data=_sparse_coo_indices_values_shape())
def test_sparse_coo(
    sparse_data,
    as_variable,
    with_out,
    native_array,
    fw,
):
    coo_ind, val_dtype, val, shp = sparse_data
    helpers.test_method(
        input_dtypes_init=["int64", val_dtype],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "coo_indices": np.array(coo_ind, dtype="int64"),
            "values": np.array(val, dtype=val_dtype),
            "dense_shape": shp,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=False,
        all_as_kwargs_np_method={},
        class_name="SparseArray",
        method_name="to_dense_array",
    )


# csr - to_dense_array
@handle_cmd_line_args
@given(sparse_data=_sparse_csr_indices_values_shape())
def test_sparse_csr(
    sparse_data,
    as_variable,
    with_out,
    native_array,
    fw,
):
    crow_indices, col_indices, value_dtype, values, shape = sparse_data
    helpers.test_method(
        input_dtypes_init=["int64", "int64", value_dtype],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "csr_crow_indices": np.array(crow_indices, dtype="int64"),
            "csr_col_indices": np.array(col_indices, dtype="int64"),
            "values": np.array(values, dtype=value_dtype),
            "dense_shape": shape,
        },
        input_dtypes_method=[],
        as_variable_flags_method=[],
        num_positional_args_method=0,
        native_array_flags_method=[],
        container_flags_method=False,
        all_as_kwargs_np_method={},
        class_name="SparseArray",
        method_name="to_dense_array",
    )


# sinc
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="sinc"),
)
def test_sinc(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sinc",
        x=x[0],
    )


# vorbis_window
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float",full=False), 
        min_num_dims=1, max_num_dims=1
    ),
    num_positional_args=helpers.num_positional_args(fn_name="vorbis_window"),
)
def test_vorbis_window(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vorbis_window",
        x=x[0],
        dtype=input_dtype,
    )


# flatten
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(
            helpers.get_shape(min_num_dims=1, max_num_dims=5), key="flatten_shape"
        ),
        min_value=-100,
        max_value=100,
    ),
    axes=helpers.get_axis(
        shape=st.shared(
            helpers.get_shape(min_num_dims=1, max_num_dims=5), key="flatten_shape"
        ),
        allow_neg=True,
        sorted=True,
        min_size=2,
        max_size=2,
        unique=False,
        force_tuple=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="flatten"),
)
def test_flatten(
    dtype_and_x,
    axes,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtypes, x = dtype_and_x
    x = np.asarray(x[0], dtype=input_dtypes[0])

    if axes[1] == 0:
        start_dim, end_dim = axes[1], axes[0]
    elif axes[0] * axes[1] < 0:
        if x.ndim + min(axes) >= max(axes):
            start_dim, end_dim = max(axes), min(axes)
        else:
            start_dim, end_dim = min(axes), max(axes)
    else:
        start_dim, end_dim = axes[0], axes[1]
    helpers.test_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=True,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="flatten",
        x=x,
        start_dim=start_dim,
        end_dim=end_dim,
    )


# lcm
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="lcm"),
)
def test_lcm(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="lcm",
        test_gradients=True,
        x1=x[0],
        x2=x[1],
    )


# hann_window
@handle_cmd_line_args
@given(
    window_length=helpers.ints(min_value=1, max_value=10),
    input_dtype=helpers.get_dtypes("integer"),
    periodic=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="hann_window"),
    dtype=helpers.get_dtypes("float"),
)
def test_hann_window(
    window_length,
    input_dtype,
    periodic,
    dtype,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="hann_window",
        window_length=window_length,
        periodic=periodic,
        dtype=dtype,
    )


@handle_cmd_line_args
@given(
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
    num_positional_args=helpers.num_positional_args(fn_name="max_pool2d"),
)
def test_max_pool2d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="max_pool2d",
        rtol_=1e-2,
        atol_=1e-2,
        ground_truth_backend="jax",
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


# kaiser_window
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=(1, 1),
        min_value=1,
        max_value=10,
    ),
    periodic=st.booleans(),
    beta=st.floats(min_value=0, max_value=5),
    dtype=helpers.get_dtypes("float"),
    num_positional_args=helpers.num_positional_args(fn_name="kaiser_window"),
)
def test_kaiser_window(
    dtype_and_x,
    periodic,
    beta,
    dtype,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="kaiser_window",
        window_length=x[0],
        periodic=periodic,
        beta=beta,
        dtype=dtype,
    )


# moveaxis
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="moveaxis"),
)
def test_moveaxis(
    dtype_and_a,
    source,
    destination,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, a = dtype_and_a
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="moveaxis",
        a=a[0],
        source=source,
        destination=destination,
    )


@handle_cmd_line_args
@given(
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


@handle_cmd_line_args
@given(
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


@st.composite
def _pad_helper(draw):
    dtype, value, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            ret_shape=True,
            min_num_dims=1,
        )
    )
    ndim = len(shape)
    pad_width = draw(
        st.one_of(
            helpers.array_values(
                dtype="int8", min_value=1, max_value=4, shape=(ndim, 2)
            ),
            helpers.ints(min_value=1, max_value=4),
        )
    )
    stat_length = draw(
        st.one_of(
            helpers.array_values(
                dtype="int8", min_value=1, max_value=4, shape=(ndim, 2)
            ),
            helpers.ints(min_value=1, max_value=4),
        )
    )
    constant_values = draw(
        st.one_of(
            helpers.array_values(dtype=dtype[0], shape=(ndim, 2)),
            helpers.array_values(dtype=dtype[0], shape=(1,)),
        )
    )
    if len(constant_values.shape) == 1:
        constant_values = constant_values[0]
    end_values = draw(
        st.one_of(
            helpers.array_values(dtype=dtype[0], shape=(ndim, 2)),
            helpers.array_values(dtype=dtype[0], shape=(1,)),
        )
    )
    if len(end_values.shape) == 1:
        end_values = end_values[0]
    dtype = dtype + 2 * ["int8"] + 2 * dtype
    return dtype, value, pad_width, stat_length, constant_values, end_values


@handle_cmd_line_args
@given(
    dtype_and_input_and_other=_pad_helper(),
    mode=st.sampled_from(
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
    ),
    reflect_type=st.sampled_from(["even", "odd"]),
    num_positional_args=helpers.num_positional_args(fn_name="pad"),
)
def test_pad(
    *,
    dtype_and_input_and_other,
    mode,
    reflect_type,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    (
        dtype,
        value,
        pad_width,
        stat_length,
        constant_values,
        end_values,
    ) = dtype_and_input_and_other
    if fw == "torch":
        assume(
            mode in ["constant", "reflect", "edge", "wrap"]
            and not np.isscalar(pad_width)
            and np.isscalar(constant_values)
        )
    elif fw == "tensorflow":
        assume(
            mode in ["constant", "reflect", "symmetric"]
            and not np.isscalar(pad_width)
            and np.isscalar(constant_values)
        )
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="pad",
        ground_truth_backend="numpy",
        x=value[0],
        pad_width=pad_width,
        mode=mode,
        stat_length=stat_length,
        constant_values=constant_values,
        end_values=end_values,
        reflect_type=reflect_type,
        out=None,
    )


# heaviside
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="heaviside"),
)
def test_heaviside(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="heaviside",
        x1=x[0],
        x2=x[0],
    )


@st.composite
def statistical_dtype_values(draw, *, function):
    large_abs_safety_factor = 2
    small_abs_safety_factor = 2
    if function in ["mean", "median", "std", "var"]:
        large_abs_safety_factor = 24
        small_abs_safety_factor = 24
    dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("float"),
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale="log",
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            valid_axis=True,
            allow_neg_axes=False,
            min_axes_size=1,
        )
    )
    shape = values[0].shape
    size = values[0].size
    max_correction = np.min(shape)
    if function == "var" or function == "std":
        if size == 1:
            correction = 0
        elif isinstance(axis, int):
            correction = draw(
                helpers.ints(min_value=0, max_value=shape[axis] - 1)
                | helpers.floats(min_value=0, max_value=shape[axis] - 1)
            )
            return dtype, values, axis, correction
        else:
            correction = draw(
                helpers.ints(min_value=0, max_value=max_correction - 1)
                | helpers.floats(min_value=0, max_value=max_correction - 1)
            )
        return dtype, values, axis, correction
    return dtype, values, axis


@handle_cmd_line_args
@given(
    dtype_x_axis=statistical_dtype_values(function="median"),
    keep_dims=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="median"),
)
def test_median(
    *,
    dtype_x_axis,
    keep_dims,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="median",
        input=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


# flipud
@handle_cmd_line_args
@given(
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="flipud"),
)
def test_flipud(
    dtype_and_m,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, m = dtype_and_m
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="flipud",
        m=m[0],
    )


# fmod
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="fmod"),
)
def test_fmod(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="fmod",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# fmax
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_nan=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="fmax"),
)
def test_fmax(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="fmax",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )
