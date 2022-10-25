"""Collection of tests for creation functions."""

# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
from ivy_tests.test_ivy.test_functional.test_core.test_dtype import astype_helper


# native_array
@handle_test(
    fn_tree="functional.ivy.native_array",
    dtype_and_x_and_cast_dtype=astype_helper(),
)
def test_native_array(
    *,
    dtype_and_x_and_cast_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    input_dtype, x, dtype = dtype_and_x_and_cast_dtype
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype,
        device=device,
    )


# linspace
@handle_test(
    dtype_and_start_stop=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e5,
        max_value=1e5,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        allow_inf=False,
        shared_dtype=True,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    fn_tree="functional.ivy.linspace",
    num=helpers.ints(min_value=1, max_value=5),
    axis=st.none(),
)
def test_linspace(
    *,
    dtype_and_start_stop,
    num,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, start_stop = dtype_and_start_stop
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=0.8,
        start=start_stop[0],
        stop=start_stop[1],
        num=num,
        axis=axis,
        device=device,
        dtype=dtype[0],
    )


# logspace
@handle_test(
    fn_tree="functional.ivy.logspace",
    dtype_and_start_stop=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=None,
        max_value=None,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        shared_dtype=True,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    num=helpers.ints(min_value=1, max_value=5),
    base=helpers.floats(min_value=0.1, max_value=3.0),
    axis=st.none(),
)
def test_logspace(
    *,
    dtype_and_start_stop,
    num,
    base,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, start_stop = dtype_and_start_stop
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        rtol_=1,  # if It's less than one it'll test for inf
        atol_=0.8,
        start=start_stop[0],
        stop=start_stop[1],
        num=num,
        base=base,
        axis=axis,
        device=device,
    )


# arange
@handle_test(
    start=helpers.ints(min_value=0, max_value=50),
    stop=helpers.ints(min_value=0, max_value=50) | st.none(),
    step=helpers.ints(min_value=-50, max_value=50).filter(
        lambda x: True if x != 0 else False
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
    fn_tree="functional.ivy.arange",
)
def test_arange(
    *,
    start,
    stop,
    step,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        start=start,
        stop=stop,
        step=step,
        dtype=dtype,
        device=device,
    )


# asarray
@handle_test(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=st.integers(min_value=1, max_value=10),
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    as_list=st.booleans(),
    fn_tree="functional.ivy.asarray",
)
def test_asarray(
    *,
    dtype_and_x,
    as_list,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, x = dtype_and_x

    if as_list:
        if isinstance(x, list):
            x = [list(i) if len(i.shape) > 0 else [float(i)] for i in x]
        else:
            x = list(x)
    else:
        if len(x) == 1:
            x = x[0]

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        object_in=x,
        dtype=dtype,
        device=device,
    )


# empty
@handle_test(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
    fn_tree="functional.ivy.empty",
)
def test_empty(
    *,
    shape,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    ret = helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
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
@handle_test(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    fn_tree="functional.ivy.empty_like",
)
def test_empty_like(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, x = dtype_and_x
    ret = helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
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
@handle_test(
    n_rows=helpers.ints(min_value=0, max_value=10),
    n_cols=st.none() | helpers.ints(min_value=0, max_value=10),
    k=helpers.ints(min_value=-10, max_value=10),
    batch_shape=st.lists(
        helpers.ints(min_value=1, max_value=10), min_size=1, max_size=2
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    fn_tree="functional.ivy.eye",
)
def test_eye(
    *,
    n_rows,
    n_cols,
    k,
    batch_shape,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        n_rows=n_rows,
        n_cols=n_cols,
        k=k,
        batch_shape=batch_shape,
        dtype=dtype,
        device=device,
    )


# from_dlpack
@handle_test(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    fn_tree="functional.ivy.from_dlpack",
)
def test_from_dlpack(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=[False],  # can't convert variables
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


@st.composite
def _fill_value(draw):
    dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
    if ivy.is_uint_dtype(dtype):
        return draw(helpers.ints(min_value=0, max_value=5))
    if ivy.is_int_dtype(dtype):
        return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


# full
@handle_test(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    fill_value=_fill_value(),
    dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
    fn_tree="functional.ivy.full",
)
def test_full(
    *,
    shape,
    fill_value,
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        shape=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        device=device,
    )


@st.composite
def _dtype_and_values(draw):
    return draw(
        helpers.dtype_and_values(
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
            dtype=draw(helpers.get_dtypes("numeric", full=False, key="dtype")),
        )
    )


# full_like
@handle_test(
    dtype_and_x=_dtype_and_values(),
    fill_value=_fill_value(),
    fn_tree="functional.ivy.full_like",
)
def test_full_like(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
    fill_value,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        fill_value=fill_value,
        dtype=dtype[0],
        device=device,
    )


# meshgrid
@handle_test(
    dtype_and_arrays=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=st.integers(min_value=2, max_value=5),
        min_num_dims=1,
        max_num_dims=1,
        shared_dtype=True,
    ),
    sparse=st.booleans(),
    indexing=st.sampled_from(["xy", "ij"]),
    fn_tree="functional.ivy.meshgrid",
)
def test_meshgrid(
    *,
    dtype_and_arrays,
    sparse,
    indexing,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, arrays = dtype_and_arrays
    kw = {}
    i = 0
    for x_ in arrays:
        kw["x{}".format(i)] = x_
        i += 1

    num_positional_args = len(arrays)

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        **kw,
        sparse=sparse,
        indexing=indexing,
    )


# ones
@handle_test(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
    fn_tree="functional.ivy.ones",
)
def test_ones(
    *,
    shape,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        shape=shape,
        dtype=dtype,
        device=device,
    )


# ones_like
@handle_test(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    fn_tree="functional.ivy.ones_like",
)
def test_ones_like(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype,
        device=device,
    )


# tril
@handle_test(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    fn_tree="functional.ivy.tril",
)
def test_tril(
    *,
    dtype_and_x,
    k,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        k=k,
    )


# triu
@handle_test(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    fn_tree="functional.ivy.triu",
)
def test_triu(
    *,
    dtype_and_x,
    k,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        k=k,
    )


# zeros
@handle_test(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    dtype=helpers.get_dtypes("integer", full=False),
    fn_tree="functional.ivy.zeros",
)
def test_zeros(
    *,
    shape,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        shape=shape,
        dtype=dtype,
        device=device,
    )


# zeros_like
@handle_test(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    fn_tree="functional.ivy.zeros_like",
)
def test_zeros_like(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype,
        device=device,
    )


# copy array
@handle_test(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    fn_tree="functional.ivy.copy_array",
)
def test_copy_array(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    dtype, x = dtype_and_x
    # smoke test
    x = ivy.array(x[0], dtype=dtype[0], device=device)
    ret = ivy.copy_array(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    helpers.assert_all_close(ivy.to_numpy(ret), ivy.to_numpy(x))
    assert id(x) != id(ret)


@st.composite
def _dtype_indices_depth_axis(draw):
    depth = draw(helpers.ints(min_value=2, max_value=100))
    dtype, indices, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            min_value=0,
            max_value=depth - 1,
            small_abs_safety_factor=4,
            ret_shape=True,
        )
    )

    axis = draw(st.integers(min_value=-1, max_value=len(shape) - 1))
    return dtype, indices, depth, axis


@st.composite
def _on_off_dtype(draw):
    dtype, value = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"), shape=(2,)
        )
    )
    [on_value, off_value] = value[0]
    return on_value, off_value, dtype[0]


# one_hot
@handle_test(
    dtype_indices_depth_axis=_dtype_indices_depth_axis(),
    on_off_dtype=_on_off_dtype(),
    fn_tree="functional.ivy.one_hot",
)
def test_one_hot(
    dtype_indices_depth_axis,
    on_off_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    device,
):
    input_dtype, indices, depth, axis = dtype_indices_depth_axis
    on_value, off_value, dtype = on_off_dtype

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        indices=indices[0],
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=axis,
        dtype=dtype,
    )
