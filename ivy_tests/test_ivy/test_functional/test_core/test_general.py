"""Collection of tests for unified general functions."""

# global
import time
import math
from types import SimpleNamespace

import pytest
from hypothesis import given, assume, strategies as st
import numpy as np
from collections.abc import Sequence

# local
import threading
import ivy

import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import (
    handle_test,
    BackendHandler,
    handle_example,
)
from ivy_tests.test_ivy.helpers.assertions import assert_all_close
from ivy_tests.test_ivy.test_functional.test_core.test_elementwise import pow_helper

try:
    import tensorflow as tf
except ImportError:
    tf = SimpleNamespace()
    tf.__version__ = None


try:
    import jax.numpy as jnp
except ImportError:
    jnp = SimpleNamespace()

try:
    import torch.multiprocessing as multiprocessing
except ImportError:
    multiprocessing = SimpleNamespace()

try:
    import ivy.functional.backends.jax
except ImportError:
    ivy.functional.backends.jax = SimpleNamespace()

try:
    import ivy.functional.backends.tensorflow
except ImportError:
    ivy.functional.backends.tensorflow = SimpleNamespace()

try:
    import ivy.functional.backends.torch
except ImportError:
    ivy.functional.backends.torch = SimpleNamespace()


# --- Helpers --- #
# --------------- #


def _composition_1():
    return ivy.relu().argmax()


def _composition_2():
    return ivy.ceil() or ivy.linspace()


def _fn1(x, y):
    return ivy.matmul(x, y)


def _fn2(x, y):
    return ivy.vecdot(x, y)


def _fn3(x, y):
    ivy.add(x, y)


def _get_shape_of_list(lst, shape=()):
    if not lst:
        return []
    if not isinstance(lst, Sequence):
        return shape
    if isinstance(lst[0], Sequence):
        length = len(lst[0])
        if not all(len(item) == length for item in lst):
            msg = "not all lists have the same length"
            raise ValueError(msg)
    shape += (len(lst),)
    shape = _get_shape_of_list(lst[0], shape)
    return shape


@st.composite  # ToDo remove when helpers.get_dtypes supports it
def _get_valid_numeric_no_unsigned(draw):
    return list(
        set(draw(helpers.get_dtypes("numeric"))).difference(
            draw(helpers.get_dtypes("unsigned"))
        )
    )


@st.composite
def _isin_data_generation_helper(draw):
    assume_unique = draw(st.booleans())
    if assume_unique:
        dtype_and_x = helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=2,
            shared_dtype=True,
        ).filter(lambda x: np.array_equal(x[1][0], np.unique(x[1][0])))
    else:
        dtype_and_x = helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=2,
            shared_dtype=True,
        )
    return assume_unique, draw(dtype_and_x)


def _supports_inplace_update(ivy_backend, test_flags) -> bool:
    supports_array_inplace_update = (
        not test_flags.as_variable and ivy_backend.inplace_arrays_supported()
    )
    supports_variable_inplace_update = (
        test_flags.as_variable and ivy_backend.inplace_variables_supported()
    )
    return supports_array_inplace_update or supports_variable_inplace_update


# fourier_encode
# @given(
#     x=helpers.dtype_and_values(ivy_np.valid_float_dtypes, min_num_dims=1),
#     max_freq=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
#     num_bands=st.integers(min_value=1,max_value=100000),
#     as_variable=st.booleans(),
#     num_positional_args=st.integers(0, 3),
#     native_array=st.booleans(),
#     container=st.booleans(),
#     instance_method=st.booleans(),
# )
# def test_fourier_encode(
#     x,
#     max_freq,
#     num_bands,
#     as_variable,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     device,
#     call,
#     fw
# ):
#     # smoke test
#     dtype_x, x = x
#     dtype_max_freq, max_freq = max_freq
#     if fw == "torch" and dtype_x in ["uint16", "uint32", "uint64"]:
#         return
#     helpers.test_function(
#         dtype_x,
#         as_variable,
#         False,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "fourier_encode",
#         x=np.asarray(x, dtype=dtype_x),
#         max_freq=np.asarray(max_freq,dtype=dtype_max_freq),
#         num_bands=num_bands
#     )


@st.composite
def _values_and_ndindices(
    draw,
    *,
    array_dtypes,
    indices_dtypes=helpers.get_dtypes("integer"),
    allow_inf=False,
    x_min_value=None,
    x_max_value=None,
    min_num_dims=2,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
):
    x_dtype, x, x_shape = draw(
        helpers.dtype_and_values(
            available_dtypes=array_dtypes,
            allow_inf=allow_inf,
            ret_shape=True,
            min_value=x_min_value,
            max_value=x_max_value,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    x_dtype = x_dtype[0] if isinstance(x_dtype, (list)) else x_dtype
    x = x[0] if isinstance(x, (list)) else x
    # indices_dims defines how far into the array to index.
    indices_dims = draw(
        helpers.ints(
            min_value=1,
            max_value=len(x_shape) - 1,
        )
    )

    # num_ndindices defines the number of elements to generate.
    num_ndindices = draw(
        helpers.ints(
            min_value=1,
            max_value=x_shape[indices_dims],
        )
    )

    # updates_dims defines how far into the array to index.
    updates_dtype, updates = draw(
        helpers.dtype_and_values(
            available_dtypes=array_dtypes,
            allow_inf=allow_inf,
            shape=x_shape[indices_dims:],
            num_arrays=num_ndindices,
            shared_dtype=True,
        )
    )
    updates_dtype = (
        updates_dtype[0] if isinstance(updates_dtype, list) else updates_dtype
    )
    updates = updates[0] if isinstance(updates, list) else updates

    indices = []
    indices_dtype = draw(st.sampled_from(indices_dtypes))
    for _ in range(num_ndindices):
        nd_index = []
        for j in range(indices_dims):
            axis_index = draw(
                helpers.ints(
                    min_value=0,
                    max_value=max(0, x_shape[j] - 1),
                )
            )
            nd_index.append(axis_index)
        indices.append(nd_index)
    indices = np.array(indices)
    return [x_dtype, indices_dtype, updates_dtype], x, indices, updates


@st.composite
def _vector_norm_helper(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float", key="clip_vector_norm"),
            min_num_dims=1,
            min_value=-100,
            max_value=100,
            abs_smallest_val=1e-2,
            safety_factor_scale="log",
        )
    )
    if ivy.is_int_dtype(dtype[0]):
        max_val = ivy.iinfo(dtype[0]).max
    else:
        max_val = ivy.finfo(dtype[0]).max
    max_x = np.abs(x[0]).max()
    if max_x > 1:
        max_p = math.log(max_val) / math.log(max_x)
    else:
        max_p = math.log(max_val)
    p = draw(helpers.floats(abs_smallest_val=1e-2, min_value=-max_p, max_value=max_p))
    max_norm_val = math.log(max_val / max_x)
    max_norm = draw(
        helpers.floats(
            large_abs_safety_factor=4,
            safety_factor_scale="log",
            min_value=1e-2,
            max_value=max_norm_val,
        )
    )
    return dtype, x, max_norm, p


@st.composite
def array_and_ndindices_batch_dims(
    draw,
    *,
    array_dtypes,
    indices_dtypes=helpers.get_dtypes("integer"),
    allow_inf=False,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
):
    x_dtype, x, x_shape = draw(
        helpers.dtype_and_values(
            available_dtypes=array_dtypes,
            allow_inf=allow_inf,
            ret_shape=True,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )

    batch_dims = draw(
        helpers.ints(
            min_value=0,
            max_value=len(x_shape) - 1,
        )
    )
    # indices_dims defines how far into the array to index.
    indices_dims = draw(
        helpers.ints(
            min_value=1,
            max_value=max(1, len(x_shape) - batch_dims),
        )
    )

    batch_shape = x_shape[0:batch_dims]
    shape_var = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims - batch_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    ndindices_shape = list(batch_shape) + list(shape_var) + [indices_dims]
    ndindices = np.zeros(ndindices_shape, dtype="int32")
    if len(ndindices_shape) <= 1:
        enumerator = ndindices
    else:
        enumerator = np.zeros(ndindices_shape[0:-1], dtype="int32")
    ndindices_dtype = draw(st.sampled_from(indices_dtypes))
    for idx, _ in np.ndenumerate(enumerator):
        bounds = []
        for j in range(0, indices_dims):
            bounds.append(x_shape[j + batch_dims] - 1)
        ndindices[idx] = draw(ndindices_with_bounds(bounds=bounds))
    ndindices = np.asarray(ndindices, ndindices_dtype)
    return [x_dtype[0], ndindices_dtype], x[0], ndindices, batch_dims


@st.composite
def ndindices_with_bounds(
    draw,
    *,
    bounds,
):
    arr = []
    for i in bounds:
        x = draw(
            helpers.ints(
                min_value=0,
                max_value=max(0, i),
            )
        )
        arr.append(x)
    return arr


# --- Main --- #
# ------------ #


# all_equal
@handle_test(
    fn_tree="functional.ivy.all_equal",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=2, max_value=10),
        min_num_dims=1,
    ),
    equality_matrix=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_all_equal(
    dtypes_and_xs, equality_matrix, test_flags, backend_fw, fn_name, on_device
):
    dtypes, arrays = dtypes_and_xs
    kw = {}
    i = 0
    for x_ in arrays:
        kw[f"x{i}"] = x_
        i += 1
    test_flags.num_positional_args = len(arrays)
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        **kw,
        equality_matrix=equality_matrix,
    )


def test_arg_info():
    return


@given(
    x_n_value=st.sampled_from(
        [
            [ivy.value_is_nan, ["x", "include_infs"]],
            [ivy.clip_matrix_norm, ["x", "max_norm", "p", "out"]],
        ]
    )
)
def test_arg_names(x_n_value):
    x, value = x_n_value
    ret = ivy.arg_names(x)
    assert ret == value


# array_equal
@handle_test(
    fn_tree="functional.ivy.array_equal",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_array_equal(dtypes_and_xs, test_flags, backend_fw, fn_name, on_device):
    dtypes, arrays = dtypes_and_xs
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x0=arrays[0],
        x1=arrays[1],
    )


@handle_test(
    fn_tree="functional.ivy.assert_supports_inplace",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    ground_truth_backend="numpy",
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_assert_supports_inplace(
    x_val_and_dtypes, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = x_val_and_dtypes
    if backend_fw in ["tensorflow", "jax", "paddle"]:
        return
    assume("bfloat16" not in dtype)
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


def test_cache_fn():
    def func():
        return ivy.random_uniform()

    # return a single cached_fn and then query this
    cached_fn = ivy.cache_fn(func)
    ret0 = cached_fn()
    ret0_again = cached_fn()
    ret1 = func()

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1

    # call ivy.cache_fn repeatedly, the new cached functions
    # each use the same global dict
    ret0 = ivy.cache_fn(func)()
    ret0_again = ivy.cache_fn(func)()
    ret1 = func()

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1


def test_cache_fn_with_args():
    def func(_):
        return ivy.random_uniform()

    # return a single cached_fn and then query this
    cached_fn = ivy.cache_fn(func)
    ret0 = cached_fn(0)
    ret0_again = cached_fn(0)
    ret1 = cached_fn(1)

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1

    # call ivy.cache_fn repeatedly, the new cached functions
    # each use the same global dict
    ret0 = ivy.cache_fn(func)(0)
    ret0_again = ivy.cache_fn(func)(0)
    ret1 = ivy.cache_fn(func)(1)

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1


# clip_matrix_norm
@handle_test(
    fn_tree="functional.ivy.clip_matrix_norm",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_value=-10,
        max_value=10,
        abs_smallest_val=1e-4,
    ),
    max_norm=st.floats(min_value=0.137, max_value=1e05),
    p=st.sampled_from([1, 2, float("inf"), "fro", "nuc"]),
)
def test_clip_matrix_norm(
    dtype_x, max_norm, p, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        max_norm=max_norm,
        p=p,
    )


# clip_vector_norm
@handle_test(
    fn_tree="functional.ivy.clip_vector_norm",
    dtype_x_max_norm_p=_vector_norm_helper(),
)
def test_clip_vector_norm(
    *, dtype_x_max_norm_p, test_flags, backend_fw, fn_name, on_device
):
    dtype, x, max_norm, p = dtype_x_max_norm_p
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        max_norm=max_norm,
        p=p,
    )


# container types
def test_container_types():
    cont_types = ivy.container_types()
    assert isinstance(cont_types, list)
    for cont_type in cont_types:
        assert hasattr(cont_type, "keys")
        assert hasattr(cont_type, "values")
        assert hasattr(cont_type, "items")


# Still to Add #
# ---------------#


@given(fw=st.sampled_from(["torch", "tensorflow", "numpy", "jax"]))
def test_current_backend_str(fw):
    ivy.set_backend(fw)
    assert ivy.current_backend_str() == fw
    ivy.previous_backend()


# default
@handle_test(
    fn_tree="functional.ivy.default",
    x=st.one_of(
        st.none(),
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            allow_inf=False,
            min_num_dims=0,
            min_dim_size=2,
        ),
        st.sampled_from([lambda *args, **kwargs: None]),
    ),
    default_val=st.one_of(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            allow_inf=False,
            min_num_dims=0,
            min_dim_size=2,
        ),
        st.sampled_from([lambda *args, **kwargs: None]),
    ),
)
def test_default(x, default_val, test_flags, backend_fw):
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        with_callable = False
        if x is not None:
            if callable(x):
                with_callable = True
            else:
                x_dtype, x = x
                x = x[0].tolist() if isinstance(x, list) else x
        else:
            if callable(default_val):
                with_callable = True
            else:
                dv_dtype, default_val = default_val
                default_val = (
                    default_val[0].tolist()
                    if isinstance(default_val, list)
                    else default_val
                )

        truth_val = ivy_backend.to_native(x if x is not None else default_val)
        if with_callable:
            assert ivy_backend.default(x, default_val) == truth_val
        else:
            assert_all_close(
                np.asarray(ivy_backend.default(x, default_val)),
                np.asarray(truth_val),
                rtol=1e-3,
                atol=1e-3,
                backend=backend_fw,
                ground_truth_backend=test_flags.ground_truth_backend,
            )


# ToDo: re-add this test once ivy.get_backend is working correctly, with the returned
#  ivy handle having no dependence on the globally set ivy
# @handle_cmd_line_args
#
# def test_class_ivy_handles(device, call):
#
#     if call is helpers.np_call:
#         # Numpy is the conflicting framework being tested against
#         pytest.skip()
#
#     class ArrayGen:
#         def __init__(self, ivyh):
#             self._ivy = ivyh
#
#         def get_array(self):
#             return self._ivy.array([0.0, 1.0, 2.0], dtype="float32", device=device)
#
#     # create instance
#     ag = ArrayGen(ivy.get_backend())
#
#     # create array from array generator
#     x = ag.get_array()
#
#     # verify this is not a numpy array
#     assert not isinstance(x, np.ndarray)
#
#     # change global framework to numpy
#     ivy.set_backend("numpy")
#
#     # create another array from array generator
#     x = ag.get_array()
#
#     # verify this is not still a numpy array
#     assert not isinstance(x, np.ndarray)


# einops_rearrange
@handle_test(
    fn_tree="functional.ivy.einops_rearrange",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=False,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=2,
        max_dim_size=2,
        min_value=-1e05,
        max_value=1e05,
    ).filter(
        lambda x: (ivy.array([x[1][0]], dtype="float32").shape[2] % 2 == 0)
        and (ivy.array([x[1][0]], dtype="float32").shape[3] % 2 == 0)
        and (x[0][0] not in ["float16", "bfloat16"])
    ),
    pattern_and_axes_lengths=st.sampled_from(
        [
            ("b h w c -> b h w c", {}),
            ("b h w c -> (b h) w c", {}),
            ("b h w c -> b c h w", {}),
            ("b h w c -> h (b w) c", {}),
            ("b h w c -> b (c h w)", {}),
            ("b (h1 h) (w1 w) c -> (b h1 w1) h w c", {"h1": 2, "w1": 2}),
            ("b (h h1) (w w1) c -> b h w (c h1 w1)", {"h1": 2, "w1": 2}),
        ]
    ),
)
def test_einops_rearrange(
    dtype_x, pattern_and_axes_lengths, test_flags, backend_fw, fn_name, on_device
):
    pattern, axes_lengths = pattern_and_axes_lengths
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        pattern=pattern,
        **axes_lengths,
    )


# einops_reduce
@handle_test(
    fn_tree="functional.ivy.einops_reduce",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=False,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=2,
        max_dim_size=2,
        min_value=-1e05,
        max_value=1e05,
    ).filter(
        lambda x: (ivy.array([x[1][0]], dtype="float32").shape[2] % 2 == 0)
        and (ivy.array([x[1][0]], dtype="float32").shape[3] % 2 == 0)
        and (x[0][0] not in ["float16", "bfloat16"])
    ),
    pattern_and_axes_lengths=st.sampled_from(
        [
            ("b c (h1 h2) (w1 w2) -> b c h1 w1", {"h2": 2, "w2": 2}),
        ]
    ),
    floattypes=helpers.get_dtypes("float"),
    reduction=st.sampled_from(["min", "max", "sum", "mean", "prod"]),
)
def test_einops_reduce(
    *,
    dtype_x,
    pattern_and_axes_lengths,
    floattypes,
    reduction,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    pattern, axes_lengths = pattern_and_axes_lengths
    dtype, x = dtype_x
    if (reduction in ["mean", "prod"]) and (dtype not in floattypes):
        dtype = ["float32"]
    # torch computes min and max differently and leads to inconsistent gradients
    if backend_fw == "torch" and reduction in ["min", "max"]:
        test_flags.test_gradients = False
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        pattern=pattern,
        reduction=reduction,
        **axes_lengths,
    )


# einops_repeat
@handle_test(
    fn_tree="functional.ivy.einops_repeat",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
    ),
    pattern_and_axes_lengths=st.sampled_from(
        [
            ("h w -> h w repeat", {"repeat": 2}),
            ("h w -> (repeat h) w", {"repeat": 2}),
            ("h w -> h (repeat w)", {"repeat": 2}),
            ("h w -> (h h2) (w w2)", {"h2": 2, "w2": 2}),
            ("h w  -> w h", {}),
        ]
    ),
)
def test_einops_repeat(
    *, dtype_x, pattern_and_axes_lengths, test_flags, backend_fw, fn_name, on_device
):
    pattern, axes_lengths = pattern_and_axes_lengths
    dtype, x = dtype_x
    assume("uint16" not in dtype)
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        pattern=pattern,
        **axes_lengths,
    )


# exists
@handle_test(
    fn_tree="functional.ivy.exists",
    x=st.one_of(
        st.none(),
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            allow_inf=False,
            min_num_dims=0,
            min_dim_size=1,
        ),
        st.sampled_from([ivy.array]),
    ),
)
def test_exists(x):
    if x is not None:
        if not callable(x):
            dtype, x = x
    ret = ivy.exists(x)
    assert isinstance(ret, bool)
    y_true = x is not None
    assert ret == y_true


def test_explicit_ivy_framework_handles(backend_fw):
    if backend_fw == "numpy":
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    # set with explicit handle caught
    ivy_exp = ivy.with_backend(backend_fw)
    assert ivy_exp.current_backend_str() == backend_fw

    # assert backend implemented function is accessible
    assert "array" in ivy_exp.__dict__
    assert callable(ivy_exp.array)

    # assert joint implemented function is also accessible
    assert "cache_fn" in ivy_exp.__dict__
    assert callable(ivy_exp.cache_fn)

    # set global ivy to numpy
    ivy.set_backend("numpy")

    # assert the explicit handle is still unchanged
    assert ivy.current_backend_str() == "numpy"
    assert ivy_exp.current_backend_str() == backend_fw

    # unset global ivy from numpy
    ivy.previous_backend()


def test_framework_setting_with_multiprocessing(backend_fw):
    if backend_fw == "numpy":
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    def worker_fn(out_queue):
        ivy.set_backend("numpy")
        x_ = np.array([0.0, 1.0, 2.0])
        for _ in range(1000):
            try:
                ivy.mean(x_)
            except TypeError:
                out_queue.put(False)
                return
        ivy.previous_backend()
        out_queue.put(True)

    # get original framework string and array
    ivy.set_backend(backend_fw)
    x = ivy.array([0.0, 1.0, 2.0])

    # start numpy loop thread
    output_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(target=worker_fn, args=(output_queue,))
    worker.start()

    # start local original framework loop
    for _ in range(1000):
        ivy.mean(x)
    ivy.previous_backend()

    worker.join()
    assert output_queue.get_nowait()


def test_framework_setting_with_threading(backend_fw):
    if backend_fw == "jax":
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    def thread_fn():
        x_ = jnp.array([0.0, 1.0, 2.0])
        ivy.set_backend("jax")
        for _ in range(2000):
            try:
                ivy.mean(x_)
            except TypeError:
                return False
        ivy.previous_backend()
        return True

    # start jax loop thread
    thread = threading.Thread(target=thread_fn)
    thread.start()
    time.sleep(0.01)
    ivy.set_backend(backend_fw)
    x = ivy.array([0.0, 1.0, 2.0])
    # start local original framework loop
    for _ in range(2000):
        ivy.mean(x)
    ivy.previous_backend()
    assert not thread.join()


# function_supported_devices_and_dtypes
@pytest.mark.parametrize(
    "func",
    [_composition_1, _composition_2],
)
def test_function_supported_device_and_dtype(func, backend_fw):
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        res = ivy_backend.function_supported_devices_and_dtypes(func, recurse=True)
        exp = {"cpu": func.test_unsupported_devices_and_dtypes.copy()["cpu"]}
        for dev in exp:
            exp[dev] = tuple(
                set(ivy.valid_dtypes).difference(exp[dev][ivy.current_backend_str()])
            )

        all_key = set(res.keys()).union(set(exp.keys()))
        for key in all_key:
            assert key in res
            assert key in exp
            assert set(res[key]) == set(exp[key])


# function_unsupported_devices_and_dtypes
@pytest.mark.parametrize(
    "func",
    [_composition_1, _composition_2],
)
def test_function_unsupported_devices(func, backend_fw):
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        res = ivy_backend.function_unsupported_devices_and_dtypes(func)
        exp = func.test_unsupported_devices_and_dtypes.copy()
        for dev in exp:
            exp[dev] = exp[dev][backend_fw]
        devs = list(exp.keys())
        for dev in devs:
            if len(exp[dev]) == 0:
                exp.pop(dev)

        all_key = set(res.keys()).union(set(exp.keys()))
        for key in all_key:
            assert key in res
            assert key in exp
            assert set(res[key]) == set(exp[key])


# gather
@handle_test(
    fn_tree="functional.ivy.gather",
    params_indices_others=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_gather(params_indices_others, test_flags, backend_fw, fn_name, on_device):
    dtypes, params, indices, axis, batch_dims = params_indices_others
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        xs_grad_idxs=[[0, 0]],
        params=params,
        indices=indices,
        axis=axis,
        batch_dims=batch_dims,
    )


# gather_nd
@handle_test(
    fn_tree="functional.ivy.gather_nd",
    params_n_ndindices_batch_dims=array_and_ndindices_batch_dims(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        allow_inf=False,
    ),
)
def test_gather_nd(
    params_n_ndindices_batch_dims, test_flags, backend_fw, fn_name, on_device
):
    dtypes, params, ndindices, batch_dims = params_n_ndindices_batch_dims
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        xs_grad_idxs=[[0, 0]],
        params=params,
        indices=ndindices,
        batch_dims=batch_dims,
    )


def test_get_all_arrays_in_memory():
    return


# get_item
# TODO: add container and array instance methods
@handle_test(
    fn_tree="functional.ivy.get_item",
    ground_truth_backend="numpy",
    dtypes_x_query=helpers.dtype_array_query(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    copy=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
    container_flags=st.just([False]),
    test_with_copy=st.just(True),
)
@handle_example(
    test_example=True,
    test_flags={
        "num_positional_args": 2,
    },
    dtypes_x_query=(["float32", "bool"], np.ones((1, 3, 3)), (np.array([True]), 2, 2)),
    copy=None,
    fn_name="get_item",
)
def test_get_item(
    dtypes_x_query,
    copy,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtypes, x, query = dtypes_x_query
    try:
        helpers.test_function(
            input_dtypes=dtypes,
            test_flags=test_flags,
            on_device=on_device,
            backend_to_test=backend_fw,
            fn_name=fn_name,
            x=x,
            query=query,
            copy=copy,
        )
    except ivy.utils.exceptions.IvyBackendException as e:
        if backend_fw == "paddle" and "only supports access to dimension 0 to 9" in e:
            assume(False)
        else:
            raise


# get_min_base
def test_get_min_base():
    assert ivy.min_base == 1e-5


# get_min_denominator
def test_get_min_denominator():
    assert ivy.min_denominator == 1e-12


# get_num_dims
@handle_test(
    fn_tree="functional.ivy.get_num_dims",
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    as_array=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_get_num_dims(
    x0_n_x1_n_res, as_array, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = x0_n_x1_n_res
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        as_array=as_array,
    )


# get_queue_timeout
@given(
    x=st.floats(allow_nan=False, allow_infinity=False),
)
def test_get_queue_timeout(x):
    ivy.set_queue_timeout(x)
    ret = ivy.queue_timeout
    assert ret == x


# get_referrers_recursive
def test_get_referrers_recursive():
    class SomeClass:
        def __init__(self):
            self.x = [1, 2]
            self.y = [self.x]

    some_obj = SomeClass()
    refs = ivy.get_referrers_recursive(some_obj.x)
    ref_keys = refs.keys()
    assert len(ref_keys) == 3
    assert "repr" in ref_keys
    assert refs["repr"] == "[1,2]"
    y_id = str(id(some_obj.y))
    y_refs = refs[y_id]
    assert y_refs["repr"] == "[[1,2]]"
    some_obj_dict_id = str(id(some_obj.__dict__))
    assert y_refs[some_obj_dict_id] == "tracked"
    dict_refs = refs[some_obj_dict_id]
    assert dict_refs["repr"] == "{'x':[1,2],'y':[[1,2]]}"
    some_obj_id = str(id(some_obj))
    some_obj_refs = dict_refs[some_obj_id]
    assert some_obj_refs["repr"] == str(some_obj).replace(" ", "")
    assert len(some_obj_refs) == 1


# get_tmp_dir
def test_get_tmp_dir():
    ret = ivy.tmp_dir
    assert ret == "/tmp"


# has_nans
@handle_test(
    fn_tree="functional.ivy.has_nans",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_nan=True,
        allow_inf=True,
    ),
    include_infs=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_has_nans(
    *, x_val_and_dtypes, include_infs, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = x_val_and_dtypes
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        include_infs=include_infs,
    )


def test_inplace_arrays_supported(backend_fw):
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        if backend_fw in ["numpy", "torch"]:
            assert ivy_backend.inplace_arrays_supported()
        elif backend_fw in ["jax", "tensorflow", "paddle"]:
            assert not ivy_backend.inplace_arrays_supported()
        else:
            raise RuntimeError("Unrecognized framework")


# inplace_decrement
@handle_test(
    fn_tree="functional.ivy.inplace_decrement",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        num_arrays=2,
        shared_dtype=True,
        safety_factor_scale="log",
    ),
)
def test_inplace_decrement(x_val_and_dtypes, test_flags, on_device, backend_fw):
    dtype = x_val_and_dtypes[0][0]
    x, val = x_val_and_dtypes[1]
    x, val = x.tolist(), val.tolist()
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        x = ivy_backend.array(x, dtype=dtype, device=on_device)
        val = ivy_backend.array(val, dtype=dtype, device=on_device)
        new_val = x - val
        supports_update = _supports_inplace_update(ivy_backend, test_flags)
        if supports_update:
            x_inplace = ivy_backend.inplace_decrement(x, val)
            assert id(x_inplace) == id(x)
            x = helpers.flatten_and_to_np(ret=x, backend=backend_fw)
            new_val = helpers.flatten_and_to_np(ret=new_val, backend=backend_fw)
            helpers.value_test(
                ret_np_flat=x, ret_np_from_gt_flat=new_val, backend=backend_fw
            )


# inplace_increment
@handle_test(
    fn_tree="functional.ivy.inplace_increment",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_inplace_increment(x_val_and_dtypes, test_flags, on_device, backend_fw):
    dtype = x_val_and_dtypes[0][0]
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        if dtype in ivy_backend.function_unsupported_dtypes(
            ivy_backend.inplace_increment
        ):
            return
        x, val = x_val_and_dtypes[1]
        x, val = x.tolist(), val.tolist()
        x = ivy_backend.array(x, dtype=dtype, device=on_device)
        val = ivy_backend.array(val, dtype=dtype, device=on_device)
        new_val = x + val
        supports_update = _supports_inplace_update(ivy_backend, test_flags)
        if supports_update:
            x_inplace = ivy_backend.inplace_increment(x, val)
            assert id(x_inplace) == id(x)
            x = helpers.flatten_and_to_np(ret=x, backend=backend_fw)
            new_val = helpers.flatten_and_to_np(ret=new_val, backend=backend_fw)
            helpers.value_test(
                ret_np_flat=x, ret_np_from_gt_flat=new_val, backend=backend_fw
            )


# inplace_update
@handle_test(
    fn_tree="functional.ivy.inplace_update",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    keep_x_dtype=st.booleans(),
    inplace_mode=st.just("lenient"),
)
def test_inplace_update(
    x_val_and_dtypes, keep_x_dtype, inplace_mode, test_flags, on_device, backend_fw
):
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        dtype = x_val_and_dtypes[0][0]
        if dtype in ivy_backend.function_unsupported_dtypes(ivy_backend.inplace_update):
            return
        x, val = x_val_and_dtypes[1]
        x = ivy_backend.array(x.tolist(), dtype=dtype, device=on_device)
        val = ivy_backend.array(val.tolist(), dtype=dtype, device=on_device)

        ivy_backend.set_inplace_mode(inplace_mode)
        supports_update = _supports_inplace_update(ivy_backend, test_flags)
        if supports_update or ivy_backend.inplace_mode == "lenient":
            if keep_x_dtype:
                x_dtype = x.dtype
                x_inplace = ivy_backend.inplace_update(x, val, keep_input_dtype=True)
                assert x_dtype == x_inplace.dtype
            else:
                x_inplace = ivy_backend.inplace_update(x, val)
            assert id(x_inplace) == id(x)
            x = helpers.flatten_and_to_np(backend=backend_fw, ret=x)
            val = helpers.flatten_and_to_np(backend=backend_fw, ret=val)
            helpers.value_test(
                backend=backend_fw, ret_np_flat=x, ret_np_from_gt_flat=val
            )
        elif not supports_update and ivy_backend.inplace_mode == "strict":
            with pytest.raises(ivy.utils.exceptions.InplaceUpdateException):
                ivy_backend.inplace_update(x, val)


def test_inplace_variables_supported(backend_fw):
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        if backend_fw in ["numpy", "torch", "tensorflow"]:
            assert ivy_backend.inplace_variables_supported()
        elif backend_fw in ["jax", "paddle"]:
            assert not ivy_backend.inplace_variables_supported()
        else:
            raise RuntimeError("Unrecognized framework")


# is_array
@handle_test(
    fn_tree="functional.ivy.is_array",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    exclusive=st.booleans(),
    as_variable_flags=st.just([False]),
    container_flags=st.just([False]),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_is_array(
    x_val_and_dtypes, exclusive, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = x_val_and_dtypes
    # as_variable=False as the result can't be consistent across backends
    if test_flags.container[0]:
        # container instance methods should also not be tested
        test_flags.instance_method = False
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        exclusive=exclusive,
    )


# is_ivy_array
@handle_test(
    fn_tree="functional.ivy.is_ivy_array",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    exclusive=st.booleans(),
    ground_truth_backend="numpy",
    as_variable_flags=st.just([False]),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_is_ivy_array(
    *, x_val_and_dtypes, exclusive, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = x_val_and_dtypes
    # as_variable=False as the result can't be consistent across backends
    if test_flags.container[0]:
        # container instance methods should also not be tested
        test_flags.instance_method = False
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        exclusive=exclusive,
    )


# is_ivy_container
@handle_test(
    fn_tree="functional.ivy.is_ivy_container",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    test_with_out=st.just(False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_is_ivy_container(x_val_and_dtypes, test_flags, backend_fw, fn_name, on_device):
    dtype, x = x_val_and_dtypes
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# is_native_array
@handle_test(
    fn_tree="functional.ivy.is_native_array",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    exclusive=st.booleans(),
    as_variable_flags=st.just([False]),
    container_flags=st.just([False]),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_is_native_array(
    *, x_val_and_dtypes, test_flags, exclusive, backend_fw, fn_name, on_device
):
    dtype, x = x_val_and_dtypes
    # as_variable=False as the result can't be consistent across backends
    if test_flags.container[0]:
        # container instance methods should also not be tested
        test_flags.instance_method = False
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        exclusive=exclusive,
    )


@handle_test(
    fn_tree="functional.ivy.isin",
    assume_unique_and_dtype_and_x=_isin_data_generation_helper(),
    invert=st.booleans(),
    ground_truth_backend="numpy",
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_isin(
    assume_unique_and_dtype_and_x,
    invert,
    test_flags,
    backend_fw,
    on_device,
):
    assume_unique, x_and_dtype = assume_unique_and_dtype_and_x
    dtypes, values = x_and_dtype
    elements, test_elements = values
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name="isin",
        elements=elements,
        test_elements=test_elements,
        invert=invert,
        assume_unique=assume_unique,
    )


@handle_test(
    fn_tree="functional.ivy.itemsize",
    x_and_dtype=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    ground_truth_backend="numpy",
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_itemsize(x_and_dtype, test_flags, backend_fw, fn_name, on_device):
    dtype, x = x_and_dtype
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# match_kwargs
@given(allow_duplicates=st.booleans())
def test_match_kwargs(allow_duplicates):
    def func_a(a, b, c=2):
        pass

    def func_b(a, d, e=5):
        return None

    class ClassA:
        def __init__(self, c, f, g=3):
            pass

    kwargs = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6}
    kwfa, kwfb, kwca = ivy.match_kwargs(
        kwargs, func_a, func_b, ClassA, allow_duplicates=allow_duplicates
    )
    if allow_duplicates:
        assert kwfa == {"a": 0, "b": 1, "c": 2}
        assert kwfb == {"a": 0, "d": 3, "e": 4}
        assert kwca == {"c": 2, "f": 5, "g": 6}
    else:
        assert kwfa == {"a": 0, "b": 1, "c": 2}
        assert kwfb == {"d": 3, "e": 4}
        assert kwca == {"f": 5, "g": 6}


def test_num_arrays_in_memory():
    return


def test_print_all_arrays_in_memory():
    return


# scatter_flat
@handle_test(
    fn_tree="functional.ivy.scatter_flat",
    x=st.integers(min_value=1, max_value=10).flatmap(
        lambda n: st.tuples(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                min_num_dims=1,
                max_num_dims=1,
                min_dim_size=n,
                max_dim_size=n,
            ),
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("integer"),
                min_value=0,
                max_value=max(n - 1, 0),
                min_num_dims=1,
                max_num_dims=1,
                min_dim_size=n,
                max_dim_size=n,
            ).filter(lambda d_n_v: len(set(d_n_v[1][0])) == len(d_n_v[1][0])),
            st.integers(min_value=n, max_value=n),
        )
    ),
    reduction=st.sampled_from(["sum", "min", "max", "replace"]),
    ground_truth_backend="tensorflow",
)
def test_scatter_flat(x, reduction, test_flags, backend_fw, fn_name, on_device):
    # scatter_flat throws an error while computing gradients for tensorflow
    # this has been fixed in the newer versions of tensorflow (2.10.0 onwards)
    if backend_fw == "tensorflow":
        grad_support_version = [2, 10, 0]
        k = 0
        for number in [int(s) for s in tf.__version__.split(".") if s.isdigit()]:
            if k > len(grad_support_version):
                break
            if number < grad_support_version[k]:
                test_flags.test_gradients = False
            k += 1
    (val_dtype, vals), (ind_dtype, ind), size = x
    helpers.test_function(
        input_dtypes=ind_dtype + val_dtype,
        test_flags=test_flags,
        xs_grad_idxs=[[0, 1]],
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        indices=ind[0],
        updates=vals[0],
        size=size,
        reduction=reduction,
    )


# scatter_nd
@handle_test(
    fn_tree="functional.ivy.scatter_nd",
    x=_values_and_ndindices(
        # ToDo: needs support for boolean arrays
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        x_min_value=0,
        x_max_value=0,
        min_num_dims=2,
        allow_inf=False,
    ),
    reduction=st.sampled_from(["sum", "min", "max", "replace"]),
    test_gradients=st.just(False),
)
def test_scatter_nd(x, reduction, test_flags, backend_fw, fn_name, on_device):
    (val_dtype, ind_dtype, update_dtype), vals, ind, updates = x
    shape = vals.shape
    helpers.test_function(
        input_dtypes=[ind_dtype, update_dtype],
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        indices=np.asarray(ind, dtype=ind_dtype),
        updates=updates,
        shape=shape,
        reduction=reduction,
    )


# Tests #
# ------#


@pytest.mark.parametrize("mode", ["lenient", "strict"])
def test_set_inplace_mode(mode):
    ivy.set_inplace_mode(mode)
    assert ivy.inplace_mode == mode


# set_item
# TODO: add container and array instance methods
@handle_test(
    fn_tree="functional.ivy.set_item",
    ground_truth_backend="numpy",
    dtypes_x_query_val=helpers.dtype_array_query_val(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    copy=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
    container_flags=st.just([False]),
    test_with_copy=st.just(True),
)
@handle_example(
    test_example=True,
    test_flags={
        "num_positional_args": 3,
    },
    dtypes_x_query_val=(
        ["int32", "int32"],
        np.ones((1, 3, 3, 3)),
        (slice(None, None, None), slice(None, None, None), slice(None, None, None), 1),
        np.zeros((3, 1)),
    ),
    copy=False,
    fn_name="set_item",
)
def test_set_item(
    dtypes_x_query_val,
    copy,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtypes, x, query, val = dtypes_x_query_val
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x,
        query=query,
        val=val,
        copy=copy,
    )


# set_min_base
@given(x=st.floats(allow_nan=False, allow_infinity=False))
def test_set_min_base(x):
    ivy.set_min_base(x)
    assert ivy.min_base == x


# set_min_denominator
@given(x=st.floats(allow_nan=False, allow_infinity=False))
def test_set_min_denominator(x):
    ivy.set_min_denominator(x)
    assert ivy.min_denominator == x


# set_queue_timeout
@given(
    x=st.floats(allow_nan=False, allow_infinity=False),
)
def test_set_queue_timeout(x):
    ivy.set_queue_timeout(x)
    ret = ivy.queue_timeout
    assert ret == x


# set_tmp_dir
def test_set_tmp_dir():
    ivy.set_tmp_dir("/new_dir")
    ret = ivy.tmp_dir
    assert ret == "/new_dir"


# shape
# TODO: add container and array methods
@handle_test(
    fn_tree="functional.ivy.shape",
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    as_array=st.booleans(),
    test_with_out=st.just(False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_shape(x0_n_x1_n_res, as_array, test_flags, backend_fw, fn_name, on_device):
    dtype, x = x0_n_x1_n_res
    # instance_method=False because the shape property would overwrite the shape method
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        as_array=as_array,
    )


# size
@handle_test(
    fn_tree="functional.ivy.size",
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_size(dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# stable_divide
@handle_test(
    fn_tree="functional.ivy.stable_divide",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=3,
        shared_dtype=True,
        small_abs_safety_factor=8,
        large_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_stable_divide(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        numerator=x[0],
        denominator=x[1],
        min_denominator=x[2],
    )


# stable_pow
@handle_test(
    fn_tree="functional.ivy.stable_pow",
    dtypes_and_xs=pow_helper(available_dtypes=_get_valid_numeric_no_unsigned()),
    min_base=helpers.floats(
        min_value=0, max_value=1, small_abs_safety_factor=8, safety_factor_scale="log"
    ),
    test_with_out=st.just(False),
)
def test_stable_pow(
    *, dtypes_and_xs, min_base, test_flags, backend_fw, fn_name, on_device
):
    dtypes, xs = dtypes_and_xs
    assume(all("bfloat16" not in x for x in dtypes))
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=1e-1,
        base=xs[0][0],
        exponent=np.abs(xs[1]),
        min_base=min_base,
    )


@handle_test(
    fn_tree="functional.ivy.strides",
    x_and_dtype=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_strides(x_and_dtype, test_flags, backend_fw, fn_name, on_device):
    dtype, x = x_and_dtype
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


@handle_test(
    fn_tree="functional.ivy.supports_inplace_updates",
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_supports_inplace_updates(
    x_val_and_dtypes, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = x_val_and_dtypes
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        test_values=False,
        x=x[0],
    )


# to_list
@handle_test(
    fn_tree="functional.ivy.to_list",
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        large_abs_safety_factor=20,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_to_list(x0_n_x1_n_res, test_flags, backend_fw, fn_name, on_device):
    dtype, x = x0_n_x1_n_res
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# to_numpy
@handle_test(
    fn_tree="functional.ivy.to_numpy",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    copy=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    test_with_copy=st.just(True),
)
def test_to_numpy(*, dtype_x, copy, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    # torch throws an exception
    if ivy.current_backend_str() == "torch" and not copy:
        return
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        copy=copy,
    )


# to_scalar
@handle_test(
    fn_tree="functional.ivy.to_scalar",
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
        large_abs_safety_factor=20,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    test_with_copy=st.just(True),
)
def test_to_scalar(x0_n_x1_n_res, test_flags, backend_fw, fn_name, on_device):
    dtype, x = x0_n_x1_n_res
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# try_else_none
@given(
    x=st.booleans(),
)
def test_try_else_none(x):
    if x:
        fn = ivy.try_else_none(lambda: True)
        assert fn() is True
    else:
        fn = ivy.try_else_none(lambda x: x)
        assert fn is None


@pytest.mark.parametrize("mode", ["lenient", "strict"])
def test_unset_inplace_mode(mode):
    ivy.set_inplace_mode(mode)
    ivy.unset_inplace_mode()
    assert ivy.inplace_mode == "lenient"


def test_use_within_use_framework():
    with ivy.functional.backends.numpy.use:
        pass
    with ivy.functional.backends.jax.use:
        pass
    with ivy.functional.backends.tensorflow.use:
        pass
    with ivy.functional.backends.torch.use:
        pass


# value_is_nan
@handle_test(
    fn_tree="functional.ivy.value_is_nan",
    val_dtype=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_dim_size=1,
        max_num_dims=1,
        allow_nan=True,
        allow_inf=True,
    ),
    include_infs=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_value_is_nan(
    *, val_dtype, include_infs, test_flags, backend_fw, fn_name, on_device
):
    dtype, val = val_dtype
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=val[0],
        include_infs=include_infs,
    )


# vmap
@handle_test(
    fn_tree="functional.ivy.vmap",
    func=st.sampled_from([_fn1, _fn2, _fn3]),
    dtype_and_arrays_and_axes=helpers.arrays_and_axes(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        num=2,
        return_dtype=True,
    ),
    in_axes_as_cont=st.booleans(),
)
def test_vmap(func, dtype_and_arrays_and_axes, in_axes_as_cont, backend_fw):
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        dtype, generated_arrays, in_axes = dtype_and_arrays_and_axes
        arrays = [ivy_backend.native_array(array) for array in generated_arrays]
        assume(
            ivy_backend.as_ivy_dtype(dtype[0])
            not in ivy_backend.function_unsupported_dtypes(ivy_backend.vmap)
        )

        if in_axes_as_cont:
            vmapped_func = ivy_backend.vmap(func, in_axes=in_axes, out_axes=0)
        else:
            vmapped_func = ivy_backend.vmap(func, in_axes=0, out_axes=0)

        assert callable(vmapped_func)

        try:
            fw_res = helpers.flatten_and_to_np(
                ret=vmapped_func(*arrays), backend=backend_fw
            )
            fw_res = fw_res if len(fw_res) else None
        except Exception:
            fw_res = None

    with BackendHandler.update_backend("jax") as gt_backend:
        arrays = [gt_backend.native_array(array) for array in generated_arrays]
        if in_axes_as_cont:
            jax_vmapped_func = gt_backend.vmap(func, in_axes=in_axes, out_axes=0)
        else:
            jax_vmapped_func = gt_backend.vmap(func, in_axes=0, out_axes=0)

        assert callable(jax_vmapped_func)

        try:
            jax_res = helpers.flatten_and_to_np(
                ret=jax_vmapped_func(*arrays), backend="jax"
            )
            jax_res = jax_res if len(jax_res) else None
        except Exception:
            jax_res = None

    if fw_res is not None and jax_res is not None:
        helpers.value_test(
            backend=backend_fw,
            ground_truth_backend="jax",
            ret_np_flat=fw_res,
            ret_np_from_gt_flat=jax_res,
            rtol=1e-1,
            atol=1e-1,
        )

    elif fw_res is None and jax_res is None:
        pass
    else:
        assert False, "One of the results is None while other isn't"


_composition_1.test_unsupported_devices_and_dtypes = {
    "cpu": {
        "numpy": ("bfloat16",),
        "jax": ("complex64", "complex128"),
        "tensorflow": ("complex64", "complex128"),
        "torch": (
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "complex64",
            "complex128",
        ),
        "paddle": ("uint16", "uint32", "uint64", "bfloat16", "complex64", "complex128"),
    },
    "gpu": {
        "numpy": ivy.all_dtypes,
        "jax": ("complex64", "complex128"),
        "tensorflow": ("complex64", "complex128"),
        "torch": ("complex64", "float16", "uint16", "complex128", "uint64", "uint32"),
        "paddle": ivy.all_dtypes,
    },
    "tpu": {
        "numpy": ivy.all_dtypes,
        "jax": ivy.all_dtypes,
        "tensorflow": ivy.all_dtypes,
        "torch": ivy.all_dtypes,
        "paddle": ivy.all_dtypes,
    },
}
_composition_2.test_unsupported_devices_and_dtypes = {
    "cpu": {
        "numpy": ("bfloat16", "complex64", "complex128"),
        "jax": ("complex64", "complex128"),
        "tensorflow": ("complex64", "complex128"),
        "torch": ("uint16", "uint32", "uint64", "float16", "complex64", "complex128"),
        "paddle": (
            "uint16",
            "uint32",
            "uint64",
            "bfloat16",
        ),
    },
    "gpu": {
        "numpy": ivy.all_dtypes,
        "jax": ("complex64", "complex128"),
        "tensorflow": ("complex64", "complex128"),
        "torch": ("uint16", "uint64", "uint32", "complex128", "float16", "complex64"),
        "paddle": ivy.all_dtypes,
    },
    "tpu": {
        "numpy": ivy.all_dtypes,
        "jax": ivy.all_dtypes,
        "tensorflow": ivy.all_dtypes,
        "torch": ivy.all_dtypes,
        "paddle": ivy.all_dtypes,
    },
}
