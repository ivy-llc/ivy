"""Collection of tests for creation functions."""

# global
from hypothesis import strategies as st, assume
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
from ivy_tests.test_ivy.test_functional.test_core.test_dtype import astype_helper


# native_array
@handle_test(
    fn_tree="functional.ivy.native_array",
    dtype_and_x_and_cast_dtype=astype_helper(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_native_array(
    *,
    dtype_and_x_and_cast_dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, dtype = dtype_and_x_and_cast_dtype
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# linspace
@handle_test(
    fn_tree="functional.ivy.linspace",
    dtype_and_start_stop_axis=helpers.dtype_values_axis(
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
        valid_axis=True,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False),
    num=helpers.ints(min_value=1, max_value=5),
    endpoint=st.booleans(),
)
def test_linspace(
    *,
    dtype_and_start_stop_axis,
    num,
    endpoint,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtypes, start_stop, axis = dtype_and_start_stop_axis
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=0.8,
        start=start_stop[0],
        stop=start_stop[1],
        num=num,
        axis=axis,
        endpoint=endpoint,
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# logspace
@handle_test(
    fn_tree="functional.ivy.logspace",
    dtype_and_start_stop_axis=helpers.dtype_values_axis(
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
        valid_axis=True,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False),
    num=helpers.ints(min_value=1, max_value=5),
    base=helpers.floats(min_value=0.1, max_value=20.0),
    endpoint=st.booleans(),
)
def test_logspace(
    *,
    dtype_and_start_stop_axis,
    dtype,
    num,
    base,
    endpoint,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtypes, start_stop, axis = dtype_and_start_stop_axis
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1,  # if It's less than one it'll test for inf
        atol_=0.8,
        start=start_stop[0],
        stop=start_stop[1],
        num=num,
        base=base,
        axis=axis,
        endpoint=endpoint,
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# arange
@handle_test(
    fn_tree="functional.ivy.arange",
    start=helpers.ints(min_value=0, max_value=50),
    stop=helpers.ints(min_value=0, max_value=50) | st.none(),
    step=helpers.ints(min_value=-50, max_value=50).filter(
        lambda x: True if x != 0 else False
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_arange(
    *,
    start,
    stop,
    step,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
    on_device,
):
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        start=start,
        stop=stop,
        step=step,
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


@st.composite
def _asarray_helper(draw):
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            num_arrays=st.integers(min_value=1, max_value=10),
            min_num_dims=0,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
            shared_dtype=True,
        )
    )
    dtype = draw(
        helpers.get_castable_dtype(
            draw(helpers.get_dtypes("numeric")), dtype=x_dtype[0]
        )
    )[-1]
    return x_dtype, x, dtype


# asarray
# TODO: Fix container, instance methods and as_variable
@handle_test(
    fn_tree="functional.ivy.asarray",
    x_dtype_x_and_dtype=_asarray_helper(),
    as_list=st.booleans(),
    test_gradients=st.just(False),
)
def test_asarray(
    *,
    x_dtype_x_and_dtype,
    as_list,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    x_dtype, x, dtype = x_dtype_x_and_dtype

    if as_list:
        if isinstance(x, list):
            x = [
                (
                    list(i)
                    if len(i.shape) > 0
                    else [complex(i) if "complex" in x_dtype[0] else float(i)]
                )
                for i in x
            ]
        else:
            x = list(x)
        # ToDo: remove this once the tests are able to generate a container of lists
        # than a list of containers
        assume(
            not (
                test_flags.container[0]
                or test_flags.instance_method
                or test_flags.with_out
            )
        )
    else:
        if len(x) == 1:
            x = x[0]
        else:
            # ToDo: remove this once the tests are able to generate a container of lists
            # than a list of containers
            assume(
                not (
                    test_flags.container[0]
                    or test_flags.instance_method
                    or test_flags.with_out
                )
            )

    helpers.test_function(
        input_dtypes=x_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        object_in=x,
        dtype=dtype,
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# empty
@handle_test(
    fn_tree="functional.ivy.empty",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_empty(
    *,
    shape,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    ret = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        shape=shape,
        dtype=dtype[0],
        device=on_device,
        test_values=False,
        ground_truth_backend=ground_truth_backend,
        return_flat_np_arrays=True,
    )
    helpers.assert_same_type_and_shape(ret)


# empty_like
@handle_test(
    fn_tree="functional.ivy.empty_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_gradients=st.just(False),
)
def test_empty_like(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    ret = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype[0],
        device=on_device,
        test_values=False,
        ground_truth_backend=ground_truth_backend,
        return_flat_np_arrays=True,
    )
    helpers.assert_same_type_and_shape(ret)


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
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_eye(
    *,
    n_rows,
    n_cols,
    k,
    batch_shape,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        n_rows=n_rows,
        n_cols=n_cols,
        k=k,
        batch_shape=batch_shape,
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# from_dlpack
@handle_test(
    fn_tree="functional.ivy.from_dlpack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_gradients=st.just(False),
)
def test_from_dlpack(
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
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        ground_truth_backend=ground_truth_backend,
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
    fn_tree="functional.ivy.full",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    fill_value=_fill_value(),
    dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_full(
    *,
    shape,
    fill_value,
    dtypes,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        shape=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
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
    fn_tree="functional.ivy.full_like",
    dtype_and_x=_dtype_and_values(),
    fill_value=_fill_value(),
)
def test_full_like(
    *,
    dtype_and_x,
    fill_value,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        fill_value=fill_value,
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# meshgrid
@handle_test(
    fn_tree="functional.ivy.meshgrid",
    dtype_and_arrays=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=st.integers(min_value=2, max_value=5),
        min_num_dims=1,
        max_num_dims=1,
        shared_dtype=True,
    ),
    sparse=st.booleans(),
    indexing=st.sampled_from(["xy", "ij"]),
    test_with_out=st.just(False),
)
def test_meshgrid(
    *,
    dtype_and_arrays,
    test_flags,
    sparse,
    indexing,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, arrays = dtype_and_arrays
    kw = {}
    i = 0
    for x_ in arrays:
        kw["x{}".format(i)] = x_
        i += 1
    test_flags.num_positional_args = len(arrays)
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        **kw,
        sparse=sparse,
        indexing=indexing,
        ground_truth_backend=ground_truth_backend,
    )


# ones
@handle_test(
    fn_tree="functional.ivy.ones",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_ones(
    *,
    shape,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        shape=shape,
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# ones_like
@handle_test(
    fn_tree="functional.ivy.ones_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_ones_like(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# tril
@handle_test(
    fn_tree="functional.ivy.tril",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
)
def test_tril(
    *,
    dtype_and_x,
    k,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        k=k,
        ground_truth_backend=ground_truth_backend,
    )


# triu
@handle_test(
    fn_tree="functional.ivy.triu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
)
def test_triu(
    *,
    dtype_and_x,
    k,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        k=k,
        ground_truth_backend=ground_truth_backend,
    )


# zeros
@handle_test(
    fn_tree="functional.ivy.zeros",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_zeros(
    *,
    shape,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        shape=shape,
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# zeros_like
@handle_test(
    fn_tree="functional.ivy.zeros_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
)
def test_zeros_like(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype[0],
        device=on_device,
        ground_truth_backend=ground_truth_backend,
    )


# copy array
@handle_test(
    fn_tree="functional.ivy.copy_array",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    to_ivy_array_bool=st.booleans(),
)
def test_copy_array(
    *,
    test_flags,
    dtype_and_x,
    to_ivy_array_bool,
    on_device,
):
    dtype, x = dtype_and_x
    # smoke test
    x = test_flags.apply_flags(x, dtype, on_device, 0)[0]
    test_flags.instance_method = (
        test_flags.instance_method if not test_flags.native_arrays[0] else False
    )
    if test_flags.instance_method:
        ret = x.copy_array(to_ivy_array=to_ivy_array_bool)
    else:
        ret = ivy.copy_array(x, to_ivy_array=to_ivy_array_bool)
    # type test
    test_ret = ret
    test_x = x
    if test_flags.container[0]:
        assert ivy.is_ivy_container(ret)
        test_ret = ret["a"]
        test_x = x["a"]
    if to_ivy_array_bool:
        assert ivy.is_ivy_array(test_ret)
    else:
        assert ivy.is_native_array(test_ret)
    # cardinality test
    assert test_ret.shape == test_x.shape
    # value test
    x, ret = ivy.to_ivy(x), ivy.to_ivy(ret)
    x_np, ret_np = helpers.flatten_and_to_np(ret=x), helpers.flatten_and_to_np(ret=ret)
    helpers.value_test(ret_np_flat=ret_np, ret_np_from_gt_flat=x_np)
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
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=(2,),
            safety_factor_scale="log",
        )
    )
    [on_value, off_value] = value[0]
    return on_value, off_value, dtype[0]


# one_hot
@handle_test(
    fn_tree="functional.ivy.one_hot",
    dtype_indices_depth_axis=_dtype_indices_depth_axis(),
    on_off_dtype=_on_off_dtype(),
    test_gradients=st.just(False),
)
def test_one_hot(
    dtype_indices_depth_axis,
    on_off_dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, indices, depth, axis = dtype_indices_depth_axis
    on_value, off_value, dtype = on_off_dtype
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        indices=indices[0],
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=axis,
        dtype=dtype,
        ground_truth_backend=ground_truth_backend,
    )


@st.composite
def _get_dtype_buffer_count_offset(draw):
    dtype, value = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
        )
    )
    value = np.array(value)
    length = value.size
    value = value.tobytes()

    offset = draw(helpers.ints(min_value=0, max_value=length - 1))
    count = draw(helpers.ints(min_value=-(2**30), max_value=length - offset))
    if count == 0:
        count = -1
    offset = offset * np.dtype(dtype[0]).itemsize

    return dtype, value, count, offset


@handle_test(
    fn_tree="functional.ivy.frombuffer",
    dtype_buffer_count_offset=_get_dtype_buffer_count_offset(),
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_frombuffer(
    dtype_buffer_count_offset,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, buffer, count, offset = dtype_buffer_count_offset
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        buffer=buffer,
        dtype=input_dtype[0],
        count=count,
        offset=offset,
        ground_truth_backend=ground_truth_backend,
    )


@handle_test(
    fn_tree="functional.ivy.triu_indices",
    n_rows=st.integers(min_value=0, max_value=5),
    n_cols=st.integers(min_value=0, max_value=5) | st.just(None),
    k=st.integers(min_value=-5, max_value=5),
    input_dtype=helpers.get_dtypes("integer"),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
)
def test_triu_indices(
    *,
    n_rows,
    n_cols,
    k,
    input_dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype = input_dtype
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        n_rows=n_rows,
        n_cols=n_cols,
        k=k,
    )
