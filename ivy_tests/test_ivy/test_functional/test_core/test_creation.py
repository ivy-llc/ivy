"""Collection of tests for creation functions."""

# global
from hypothesis import strategies as st, assume
import numpy as np
import ivy


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test, BackendHandler
import ivy_tests.test_ivy.helpers.globals as test_globals
from ivy_tests.test_ivy.test_functional.test_core.test_dtype import astype_helper


# --- Helpers --- #
# --------------- #


@st.composite
def _asarray_helper(draw):
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=st.integers(min_value=1, max_value=10),
            min_num_dims=0,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
            shared_dtype=True,
        )
    )
    with BackendHandler.update_backend(test_globals.CURRENT_BACKEND) as ivy_backend:
        x_list = ivy_backend.nested_map(lambda x: x.tolist(), x, shallow=False)
        sh = draw(helpers.get_shape(min_num_dims=1))
        sh = ivy_backend.Shape(sh)
    # np_array = x[0]
    # dim = draw(helpers.get_shape(min_num_dims=1))
    # nested_values = draw(
    #     helpers.create_nested_input(dim, [sh, np_array, x_list[0]])
    # )
    dtype = draw(
        helpers.get_castable_dtype(
            draw(helpers.get_dtypes("numeric")), dtype=x_dtype[0]
        )
    )[-1]
    dtype = draw(st.sampled_from([dtype]))
    x = draw(
        st.sampled_from(
            [
                x,
                x_list,
                # sh,
                # nested_values,
            ]
        )
    )
    return x_dtype, x, dtype


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
def _fill_value(draw):
    dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
    with BackendHandler.update_backend(test_globals.CURRENT_BACKEND) as ivy_backend:
        if ivy_backend.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy_backend.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


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


# --- Main --- #
# ------------ #


def is_capsule(o):
    t = type(o)
    return t.__module__ == "builtins" and t.__name__ == "PyCapsule"


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
    on_device,
):
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        start=start,
        stop=stop,
        step=step,
        dtype=dtype[0],
        device=on_device,
    )


# asarray
# TODO: Fix container, instance methods and as_variable
@handle_test(
    fn_tree="functional.ivy.asarray",
    x_dtype_x_and_dtype=_asarray_helper(),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
    test_with_copy=st.just(False),
)
def test_asarray(
    *,
    x_dtype_x_and_dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    x_dtype, x, dtype = x_dtype_x_and_dtype
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    assume(not test_flags.container[0])
    # avoid casting complex to non-complex
    if dtype is not None:
        assume(not ("complex" in x_dtype[0] and "complex" not in dtype))
    helpers.test_function(
        input_dtypes=x_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        object_in=x,
        dtype=dtype,
        device=on_device,
    )


# copy array
@handle_test(
    fn_tree="functional.ivy.copy_array",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    to_ivy_array_bool=st.booleans(),
    test_with_copy=st.just(True),
)
def test_copy_array(
    *,
    test_flags,
    dtype_and_x,
    to_ivy_array_bool,
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    # avoid enabling gradients for non-float arrays
    if test_flags.as_variable[0]:
        assume("float" in dtype[0])
    # smoke test
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        x = test_flags.apply_flags(
            x, dtype, 0, backend=backend_fw, on_device=on_device
        )[0]
        test_flags.instance_method = (
            test_flags.instance_method if not test_flags.native_arrays[0] else False
        )
        if test_flags.instance_method:
            ret = x.copy_array(to_ivy_array=to_ivy_array_bool)
        else:
            ret = ivy_backend.copy_array(x, to_ivy_array=to_ivy_array_bool)
        # type test
        test_ret = ret
        test_x = x
        if test_flags.container[0]:
            assert ivy_backend.is_ivy_container(ret)
            test_ret = ret["a"]
            test_x = x["a"]
        if to_ivy_array_bool:
            assert ivy_backend.is_ivy_array(test_ret)
        else:
            assert ivy_backend.is_native_array(test_ret)
        # cardinality test
        assert test_ret.shape == test_x.shape
        # value test
        x, ret = ivy_backend.to_ivy(x), ivy_backend.to_ivy(ret)
        x_np, ret_np = helpers.flatten_and_to_np(
            backend=backend_fw, ret=x
        ), helpers.flatten_and_to_np(backend=backend_fw, ret=ret)
        helpers.value_test(
            backend=backend_fw,
            ground_truth_backend=backend_fw,
            ret_np_flat=ret_np,
            ret_np_from_gt_flat=x_np,
        )
        assert id(x) != id(ret)


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
def test_empty(*, shape, dtype, test_flags, backend_fw, fn_name, on_device):
    ret = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        shape=shape,
        dtype=dtype[0],
        device=on_device,
        test_values=False,
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
def test_empty_like(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    ret = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype[0],
        device=on_device,
        test_values=False,
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
    *, n_rows, n_cols, k, batch_shape, dtype, test_flags, backend_fw, fn_name, on_device
):
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        n_rows=n_rows,
        n_cols=n_cols,
        k=k,
        batch_shape=batch_shape,
        dtype=dtype[0],
        device=on_device,
    )


# from_dlpack
@handle_test(
    fn_tree="functional.ivy.from_dlpack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="float", full=False, key="dtype"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_gradients=st.just(False),
)
def test_from_dlpack(*, dtype_and_x, backend_fw):
    if backend_fw == "numpy":
        return
    ivy.set_backend(backend_fw)
    input_dtype, x = dtype_and_x
    native_array = ivy.native_array(x[0])
    cap = ivy.to_dlpack(native_array)
    array = ivy.from_dlpack(cap)
    assert ivy.is_native_array(array)


@handle_test(
    fn_tree="functional.ivy.frombuffer",
    dtype_buffer_count_offset=_get_dtype_buffer_count_offset(),
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_frombuffer(
    dtype_buffer_count_offset, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, buffer, count, offset = dtype_buffer_count_offset
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        buffer=buffer,
        dtype=input_dtype[0],
        count=count,
        offset=offset,
    )


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
    dtypes=helpers.get_dtypes("valid", full=False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_full(*, shape, fill_value, dtypes, test_flags, backend_fw, fn_name, on_device):
    if dtypes[0].startswith("uint") and fill_value < 0:
        fill_value = -fill_value
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        shape=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        device=on_device,
    )


# full_like
@handle_test(
    fn_tree="functional.ivy.full_like",
    dtype_and_x=_dtype_and_values(),
    dtypes=helpers.get_dtypes("valid", full=False),
    fill_value=_fill_value(),
    test_gradients=st.just(False),
)
def test_full_like(
    *, dtype_and_x, dtypes, fill_value, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = dtype_and_x
    if dtypes[0].startswith("uint") and fill_value < 0:
        fill_value = -fill_value
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        fill_value=fill_value,
        dtype=dtype[0],
        device=on_device,
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
):
    input_dtypes, start_stop, axis = dtype_and_start_stop_axis
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
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
):
    input_dtypes, start_stop, axis = dtype_and_start_stop_axis
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
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
    *, dtype_and_arrays, test_flags, sparse, indexing, backend_fw, fn_name, on_device
):
    dtype, arrays = dtype_and_arrays
    kw = {}
    i = 0
    for x_ in arrays:
        kw[f"x{i}"] = x_
        i += 1
    test_flags.num_positional_args = len(arrays)
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        **kw,
        sparse=sparse,
        indexing=indexing,
    )


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
):
    input_dtype, x, dtype = dtype_and_x_and_cast_dtype
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype[0],
        device=on_device,
    )


# one_hot
@handle_test(
    fn_tree="functional.ivy.one_hot",
    dtype_indices_depth_axis=_dtype_indices_depth_axis(),
    on_off_dtype=_on_off_dtype(),
    test_gradients=st.just(False),
)
def test_one_hot(
    dtype_indices_depth_axis, on_off_dtype, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, indices, depth, axis = dtype_indices_depth_axis
    on_value, off_value, dtype = on_off_dtype
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        indices=indices[0],
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=axis,
        dtype=dtype,
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
def test_ones(*, shape, dtype, test_flags, backend_fw, fn_name, on_device):
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        shape=shape,
        dtype=dtype[0],
        device=on_device,
    )


# ones_like
@handle_test(
    fn_tree="functional.ivy.ones_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_ones_like(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype[0],
        device=on_device,
    )


# to_dlpack
@handle_test(
    fn_tree="functional.ivy.to_dlpack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="float", full=False, key="dtype"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_gradients=st.just(False),
)
def test_to_dlpack(*, dtype_and_x, backend_fw):
    ivy.set_backend(backend_fw)
    input_dtype, x = dtype_and_x
    native_array = ivy.native_array(x[0])
    cap = ivy.to_dlpack(native_array)
    assert is_capsule(cap)


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
def test_tril(*, dtype_and_x, k, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        k=k,
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
def test_triu(*, dtype_and_x, k, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        k=k,
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
    *, n_rows, n_cols, k, input_dtype, test_flags, backend_fw, fn_name, on_device
):
    input_dtype = input_dtype
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        n_rows=n_rows,
        n_cols=n_cols,
        k=k,
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
def test_zeros(*, shape, dtype, test_flags, backend_fw, fn_name, on_device):
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        shape=shape,
        dtype=dtype[0],
        device=on_device,
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
    test_gradients=st.just(False),
)
def test_zeros_like(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        dtype=dtype[0],
        device=on_device,
    )
