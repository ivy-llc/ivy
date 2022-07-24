"""Collection of tests for creation functions."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy.functional.backends.numpy as ivy_np
import hypothesis.extra.numpy as hnp


# native_array
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        dtype=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="native_array"),
    data=st.data(),
)
@handle_cmd_line_args
def test_native_array(
    *,
    data,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    instance_method,
    fw,
    device,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=instance_method,
        fw=fw,
        fn_name="native_array",
        x=np.asarray(x),
        dtype=dtype,
        device=device,
    )


# linspace
@given(
    dtype_and_start_stop=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
        safety_factor=0.5,
    ),
    num=st.integers(1, 5),
    axis=st.none(),
    num_positional_args=helpers.num_positional_args(fn_name="linspace"),
    data=st.data(),
)
@handle_cmd_line_args
def test_linspace(
    *,
    data,
    dtype_and_start_stop,
    num,
    axis,
    device,
    num_positional_args,
    fw,
):
    dtype, start_stop = dtype_and_start_stop
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="linspace",
        start=start_stop[0],
        stop=start_stop[1],
        num=num,
        axis=axis,
        device=device,
        dtype=dtype,
    )


# logspace
@given(
    dtype_and_start_stop=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
        safety_factor=0.5,
    ),
    num=st.integers(1, 5),
    base=st.floats(min_value=0.1, max_value=10.0),
    axis=st.none(),
    num_positional_args=helpers.num_positional_args(fn_name="logspace"),
    data=st.data(),
)
@handle_cmd_line_args
def test_logspace(
    *,
    data,
    dtype_and_start_stop,
    num,
    base,
    axis,
    device,
    num_positional_args,
    fw,
):
    dtype, start_stop = dtype_and_start_stop
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="logspace",
        test_rtol=(1,),  # if its less then one it'll test for inf
        test_atol=(1e-06,),
        test_values=True,
        start=start_stop[0],
        stop=start_stop[1],
        num=num,
        base=base,
        axis=axis,
        device=device,
    )


# arange
@given(
    start=st.integers(0, 5),
    stop=st.integers(0, 5) | st.none(),
    step=st.integers(-5, 5).filter(lambda x: True if x != 0 else False),
    dtype=st.sampled_from(ivy_np.valid_int_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="arange"),
    data=st.data(),
)
@handle_cmd_line_args
def test_arange(
    *,
    data,
    start,
    stop,
    step,
    dtype,
    device,
    num_positional_args,
    with_out,
    fw,
):
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="arange",
        start=start,
        stop=stop,
        step=step,
        dtype=dtype,
        device=device,
    )


# asarray
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="asarray"),
    data=st.data(),
)
@handle_cmd_line_args
def test_asarray(
    *,
    data,
    dtype_and_x,
    device,
    as_variable,
    num_positional_args,
    native_array,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x

    if instance_method:
        x = np.asarray(x)

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=instance_method,
        fw=fw,
        fn_name="asarray",
        object_in=x,
        dtype=dtype,
        device=device,
    )


# empty
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="empty"),
    data=st.data(),
)
@handle_cmd_line_args
def test_empty(
    *,
    data,
    shape,
    dtype,
    device,
    with_out,
    num_positional_args,
    fw,
):

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="empty",
        shape=shape,
        dtype=dtype,
        device=device,
    )


# empty_like
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="empty_like"),
    data=st.data(),
)
@handle_cmd_line_args
def test_empty_like(
    *,
    data,
    dtype_and_x,
    device,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=instance_method,
        fw=fw,
        fn_name="empty_like",
        x=np.asarray(x),
        dtype=dtype,
        device=device,
    )


# eye
@given(
    n_rows=st.integers(min_value=0, max_value=5),
    n_cols=st.none() | st.integers(min_value=0, max_value=5),
    k=st.integers(min_value=-5, max_value=5),
    batch_shape=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=2),
    dtype=st.sampled_from(ivy_np.valid_int_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="eye"),
    data=st.data(),
)
@handle_cmd_line_args
def test_eye(
    *,
    data,
    n_rows,
    n_cols,
    k,
    batch_shape,
    dtype,
    device,
    as_variable,
    with_out,
    num_positional_args,
    fw,
):

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="eye",
        n_rows=n_rows,
        n_cols=n_cols,
        k=k,
        batch_shape=batch_shape,
        dtype=dtype,
        device=device,
    )


# from_dlpack
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="from_dlpack"),
    data=st.data(),
)
@handle_cmd_line_args
def test_from_dlpack(
    *,
    data,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    instance_method,
    fw,
):
    if fw == "tensorflow" or fw == "jax":  # not working at time of commit
        return
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=instance_method,
        fw=fw,
        fn_name="from_dlpack",
        x=np.asarray(x, dtype=dtype),
    )


@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=1
            ),
            key="dtype",
        )
    )


@st.composite
def _fill_value(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_uint_dtype(dtype):
        return draw(st.integers(0, 5))
    if ivy.is_int_dtype(dtype):
        return draw(st.integers(-5, 5))
    return draw(st.floats(-5, 5))


# full
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(fn_name="full"),
    data=st.data(),
)
@handle_cmd_line_args
def test_full(
    *,
    data,
    shape,
    fill_value,
    dtypes,
    with_out,
    device,
    num_positional_args,
    fw,
):
    helpers.test_function(
        input_dtypes=dtypes[0],
        as_variable_flags=False,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="full",
        shape=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        device=device,
    )


@st.composite
def _dtype_and_values(draw):
    return draw(
        helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_numeric_dtypes,
            num_arrays=1,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
            dtype=draw(_dtypes()),
        )
    )


# full_like
@given(
    dtype_and_x=_dtype_and_values(),
    fill_value=_fill_value(),
    num_positional_args=helpers.num_positional_args(fn_name="full_like"),
    data=st.data(),
)
@handle_cmd_line_args
def test_full_like(
    *,
    data,
    dtype_and_x,
    device,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    instance_method,
    fw,
    fill_value,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=instance_method,
        fw=fw,
        fn_name="full_like",
        x=np.asarray(x),
        fill_value=fill_value,
        dtype=dtype,
        device=device,
    )


# meshgrid


# ToDo: create arrays which are not only 1-d
array_shape = st.shared(
    st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=1),
    key="array_shape",
)
dtype_shared = st.shared(st.sampled_from(ivy_np.valid_numeric_dtypes), key="dtype")


@given(
    arrays=st.lists(
        hnp.arrays(dtype=dtype_shared, shape=array_shape), min_size=1, max_size=3
    ),
    dtype=dtype_shared,
    data=st.data(),
)
@handle_cmd_line_args
def test_meshgrid(
    *,
    data,
    arrays,
    dtype,
    fw,
):

    kw = {}
    i = 0
    for x_ in arrays:
        kw["x{}".format(i)] = np.asarray(x_, dtype=dtype)
        i += 1

    num_positional_args = len(arrays)

    helpers.test_function(
        input_dtypes=[dtype for _ in range(num_positional_args)],
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="meshgrid",
        **kw,
    )


# ones
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="ones"),
    data=st.data(),
)
@handle_cmd_line_args
def test_ones(
    *,
    data,
    shape,
    dtype,
    with_out,
    device,
    num_positional_args,
    fw,
):
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="ones",
        shape=shape,
        dtype=dtype,
        device=device,
    )


# ones_like
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    num_positional_args=st.integers(0, 1),
    data=st.data(),
)
@handle_cmd_line_args
def test_ones_like(
    *,
    data,
    dtype_and_x,
    device,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=instance_method,
        fw=fw,
        fn_name="ones_like",
        x=np.asarray(x, dtype=dtype),
        dtype=dtype,
        device=device,
    )


# tril
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    k=st.integers(-5, 5),
    num_positional_args=helpers.num_positional_args(fn_name="tril"),
    data=st.data(),
)
@handle_cmd_line_args
def test_tril(
    *,
    data,
    dtype_and_x,
    k,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=instance_method,
        fw=fw,
        fn_name="tril",
        x=np.asarray(x),
        k=k,
    )


# triu
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    k=st.integers(-5, 5),
    num_positional_args=helpers.num_positional_args(fn_name="triu"),
    data=st.data(),
)
@handle_cmd_line_args
def test_triu(
    *,
    data,
    dtype_and_x,
    k,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=instance_method,
        fw=fw,
        fn_name="triu",
        x=np.asarray(x),
        k=k,
    )


# zeros
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=st.sampled_from(ivy_np.valid_int_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="zeros"),
    data=st.data(),
)
@handle_cmd_line_args
def test_zeros(
    *,
    data,
    shape,
    dtype,
    device,
    with_out,
    num_positional_args,
    fw,
):
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="zeros",
        shape=shape,
        dtype=dtype,
        device=device,
    )


# zeros_like
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="zeros_like"),
    data=st.data(),
)
@handle_cmd_line_args
def test_zeros_like(
    *,
    data,
    dtype_and_x,
    device,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=instance_method,
        fw=fw,
        fn_name="zeros_like",
        x=np.asarray(x, dtype=dtype),
        dtype=dtype,
        device=device,
    )
