# For Review
"""Collection of tests for creation functions."""

# global
import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# native_array
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        dtype=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="native_array"),
)
def test_native_array(
    *,
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
@handle_cmd_line_args
@given(
    dtype_and_start_stop=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        num_arrays=2,
        min_value=None,
        max_value=None,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        allow_inf=False,
        shared_dtype=True,
        small_value_safety_factor=0.5,
        large_value_safety_factor=0.5,
    ),
    num=helpers.ints(min_value=1, max_value=5),
    axis=st.none(),
    num_positional_args=helpers.num_positional_args(fn_name="linspace"),
)
def test_linspace(
    *,
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
        start=np.asarray(start_stop[0], dtype=dtype[0]),
        stop=np.asarray(start_stop[1], dtype=dtype[1]),
        num=num,
        axis=axis,
        device=device,
        dtype=dtype[0],
    )


# logspace
@handle_cmd_line_args
@given(
    dtype_and_start_stop=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        num_arrays=2,
        min_value=None,
        max_value=None,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        shared_dtype=True,
        small_value_safety_factor=0.5,
        large_value_safety_factor=0.5,
    ),
    num=helpers.ints(min_value=1, max_value=5),
    base=helpers.floats(min_value=0.1, max_value=10.0),
    axis=st.none(),
    num_positional_args=helpers.num_positional_args(fn_name="logspace"),
)
def test_logspace(
    *,
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
        rtol_=1,  # if It's less than one it'll test for inf
        atol_=1e-06,
        test_values=True,
        start=np.asarray(start_stop[0], dtype=dtype[0]),
        stop=np.asarray(start_stop[1], dtype=dtype[1]),
        num=num,
        base=base,
        axis=axis,
        device=device,
    )


# arange
@handle_cmd_line_args
@given(
    start=helpers.ints(min_value=0, max_value=50),
    stop=helpers.ints(min_value=0, max_value=50) | st.none(),
    step=helpers.ints(min_value=-50, max_value=50).filter(
        lambda x: True if x != 0 else False
    ),
    dtype=st.sampled_from(ivy_np.valid_int_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="arange"),
)
def test_arange(
    *,
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
@handle_cmd_line_args
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
)
def test_asarray(
    *,
    dtype_and_x,
    device,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="asarray",
        object_in=x,
        dtype=dtype,
        device=device,
    )


# empty
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="empty"),
)
def test_empty(
    *,
    shape,
    dtype,
    device,
    num_positional_args,
    fw,
):
    ret = helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="empty",
        shape=shape,
        dtype=dtype,
        device=device,
        test_values=False,
    )
    if not ivy.exists(ret):
        return
    res, res_np = ret
    ivy.set_backend("tensorflow")
    assert res.shape == res_np.shape
    assert res.dtype == res_np.dtype
    ivy.unset_backend()


# empty_like
@handle_cmd_line_args
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
)
def test_empty_like(
    *,
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
    ret = helpers.test_function(
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
        test_values=False,
    )
    if not ivy.exists(ret):
        return
    res, res_np = ret
    ivy.set_backend("tensorflow")
    assert res.shape == res_np.shape
    assert res.dtype == res_np.dtype
    ivy.unset_backend()


# eye
@handle_cmd_line_args
@given(
    n_rows=helpers.ints(min_value=0, max_value=10),
    n_cols=st.none() | helpers.ints(min_value=0, max_value=10),
    k=helpers.ints(min_value=-10, max_value=10),
    batch_shape=st.lists(
        helpers.ints(min_value=1, max_value=10), min_size=1, max_size=2
    ),
    dtype=st.sampled_from(ivy_np.valid_int_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="eye"),
)
def test_eye(
    *,
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
@handle_cmd_line_args
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
)
def test_from_dlpack(
    *,
    dtype_and_x,
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
        as_variable_flags=False,  # can't convert variables
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
        return draw(helpers.ints(min_value=0, max_value=5))
    if ivy.is_int_dtype(dtype):
        return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


# full
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(fn_name="full"),
)
def test_full(
    *,
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
@handle_cmd_line_args
@given(
    dtype_and_x=_dtype_and_values(),
    fill_value=_fill_value(),
    num_positional_args=helpers.num_positional_args(fn_name="full_like"),
)
def test_full_like(
    *,
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
    st.lists(helpers.ints(min_value=1, max_value=10), min_size=1, max_size=1),
    key="array_shape",
)
dtype_shared = st.shared(st.sampled_from(ivy_np.valid_numeric_dtypes), key="dtype")


@handle_cmd_line_args
@given(
    arrays=st.lists(
        hnp.arrays(dtype=dtype_shared, shape=array_shape), min_size=1, max_size=5
    ),
    indexing=st.sampled_from(["xy", "ij"]),
    dtype=dtype_shared,
)
def test_meshgrid(
    *,
    arrays,
    indexing,
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
        indexing=indexing,
    )


# ones
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="ones"),
)
def test_ones(
    *,
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
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="ones_like"),
)
def test_ones_like(
    *,
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
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    num_positional_args=helpers.num_positional_args(fn_name="tril"),
)
def test_tril(
    *,
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
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    num_positional_args=helpers.num_positional_args(fn_name="triu"),
)
def test_triu(
    *,
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
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    dtype=st.sampled_from(ivy_np.valid_int_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="zeros"),
)
def test_zeros(
    *,
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
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="zeros_like"),
)
def test_zeros_like(
    *,
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
