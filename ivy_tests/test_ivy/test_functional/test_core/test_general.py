"""Collection of tests for unified general functions."""

# global
import time
import jax.numpy as jnp
import pytest
from hypothesis import given, assume, strategies as st
import numpy as np
from collections.abc import Sequence
import torch.multiprocessing as multiprocessing

# local
import threading
import ivy
import ivy.functional.backends.jax
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
from ivy_tests.test_ivy.helpers.assertions import assert_all_close
from ivy_tests.test_ivy.test_functional.test_core.test_elementwise import pow_helper

# Helpers #
# --------#


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


# Tests #
# ------#

# set_framework
@handle_cmd_line_args
@given(fw_str=st.sampled_from(["numpy", "jax", "torch", "tensorflow"]))
def test_set_framework(fw_str, device):
    ivy.set_backend(fw_str)
    ivy.unset_backend()


# use_framework
@handle_cmd_line_args
def test_use_within_use_framework(device):
    with ivy.functional.backends.numpy.use:
        pass
    with ivy.functional.backends.jax.use:
        pass
    with ivy.functional.backends.tensorflow.use:
        pass
    with ivy.functional.backends.torch.use:
        pass


@handle_cmd_line_args
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


@handle_cmd_line_args
def test_get_referrers_recursive(device):
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


# array_equal
@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="array_equal"),
)
def test_array_equal(
    dtypes_and_xs,
    num_positional_args,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    dtypes, arrays = dtypes_and_xs
    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="array_equal",
        x0=arrays[0],
        x1=arrays[1],
    )


@st.composite
def array_and_boolean_mask(
    draw,
    *,
    array_dtypes,
    allow_inf=False,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
):
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=array_dtypes,
            allow_inf=allow_inf,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    boolean_mask_dtype, boolean_mask = draw(
        helpers.dtype_and_values(
            dtype=["bool"],
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    return [x_dtype[0], boolean_mask_dtype[0]], x[0], boolean_mask[0]


@handle_cmd_line_args
@given(
    dtype_x_indices=st.one_of(
        helpers.array_indices_axis(
            array_dtypes=helpers.get_dtypes("valid"),
            indices_dtypes=helpers.get_dtypes("integer"),
            disable_random_axis=True,
            first_dimension_only=True,
        ),
        array_and_boolean_mask(array_dtypes=helpers.get_dtypes("valid")),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="get_item"),
)
def test_get_item(
    dtype_x_indices,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    device,
):
    dtypes, x, indices = dtype_x_indices
    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=fw,
        fn_name="get_item",
        x=x,
        query=indices,
    )


# to_numpy
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="to_numpy"),
    copy=st.booleans(),
)
def test_to_numpy(
    dtype_x,
    copy,
    num_positional_args,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x = dtype_x
    # torch throws an exception
    if not copy and fw == "torch":
        return
    helpers.test_function(
        input_dtypes=dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        device_=device,
        fw=fw,
        fn_name="to_numpy",
        x=x[0],
        copy=copy,
    )


# to_scalar
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
        large_abs_safety_factor=20,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="to_scalar"),
)
def test_to_scalar(
    x0_n_x1_n_res,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x = x0_n_x1_n_res
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        device_=device,
        fw=fw,
        fn_name="to_scalar",
        x=x[0],
    )


# to_list
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        large_abs_safety_factor=20,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="to_list"),
)
def test_to_list(
    x0_n_x1_n_res,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x = x0_n_x1_n_res
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        device_=device,
        fw=fw,
        fn_name="to_list",
        x=x[0],
    )


# shape
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    as_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="shape"),
)
def test_shape(
    x0_n_x1_n_res,
    as_array,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x = x0_n_x1_n_res
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="shape",
        x=x[0],
        as_array=as_array,
    )


# get_num_dims
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    as_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="get_num_dims"),
)
def test_get_num_dims(
    x0_n_x1_n_res,
    as_array,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x = x0_n_x1_n_res
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="get_num_dims",
        x=x[0],
        as_array=as_array,
    )


# clip_vector_norm
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", key="clip_vector_norm"),
        min_num_dims=1,
        large_abs_safety_factor=16,
        small_abs_safety_factor=64,
        safety_factor_scale="log",
    ),
    max_norm_n_p=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", key="clip_vector_norm"),
        num_arrays=2,
        large_abs_safety_factor=16,
        small_abs_safety_factor=64,
        safety_factor_scale="log",
        shape=(),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="clip_vector_norm"),
)
def test_clip_vector_norm(
    dtype_x,
    max_norm_n_p,
    as_variable,
    num_positional_args,
    with_out,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x = dtype_x
    max_norm, p = max_norm_n_p[1]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="clip_vector_norm",
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        max_norm=float(max_norm),
        p=float(p),
    )


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
def values_and_ndindices(
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
    return [x_dtype, indices_dtype, updates_dtype], x, indices, updates


# scatter_flat
@handle_cmd_line_args
@given(
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
            ).filter(lambda l: len(set(l[1][0])) == len(l[1][0])),
            st.integers(min_value=n, max_value=n),
        )
    ),
    reduction=st.sampled_from(["sum", "min", "max", "replace"]),
    num_positional_args=helpers.num_positional_args(fn_name="scatter_flat"),
)
def test_scatter_flat(
    x,
    reduction,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    (val_dtype, vals), (ind_dtype, ind), size = x
    helpers.test_function(
        input_dtypes=ind_dtype + val_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="scatter_flat",
        indices=ind[0],
        updates=vals[0],
        size=size,
        reduction=reduction,
    )


# scatter_nd
@handle_cmd_line_args
@given(
    x=values_and_ndindices(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        x_min_value=0,
        x_max_value=0,
        min_num_dims=2,
        allow_inf=False,
    ),
    reduction=st.sampled_from(["sum", "min", "max", "replace"]),
    num_positional_args=helpers.num_positional_args(fn_name="scatter_nd"),
)
def test_scatter_nd(
    x,
    reduction,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    (val_dtype, ind_dtype, update_dtype), vals, ind, updates = x
    shape = vals.shape
    helpers.test_function(
        input_dtypes=[ind_dtype, update_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="scatter_nd",
        indices=np.asarray(ind, dtype=ind_dtype),
        updates=updates,
        shape=shape,
        reduction=reduction,
    )


# gather
@handle_cmd_line_args
@given(
    params_indices_others=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="gather"),
)
def test_gather(
    params_indices_others,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtypes, params, indices, axis, batch_dims = params_indices_others
    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="gather",
        params=params,
        indices=indices,
        axis=axis,
        batch_dims=batch_dims,
    )


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


# gather_nd
@handle_cmd_line_args
@given(
    params_n_ndindices_batch_dims=array_and_ndindices_batch_dims(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        allow_inf=False,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="gather_nd"),
)
def test_gather_nd(
    params_n_ndindices_batch_dims,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtypes, params, ndindices, batch_dims = params_n_ndindices_batch_dims
    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="gather_nd",
        params=params,
        indices=ndindices,
        batch_dims=batch_dims,
    )


# exists
@handle_cmd_line_args
@given(
    x=st.one_of(
        st.none(),
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            allow_inf=False,
            min_num_dims=0,
            min_dim_size=1,
        ),
        st.sampled_from([ivy.array]),
    )
)
def test_exists(x):
    if x is not None:
        if not hasattr(x, "__call__"):
            dtype, x = x
    ret = ivy.exists(x)
    assert isinstance(ret, bool)
    y_true = x is not None
    assert ret == y_true


# default
@handle_cmd_line_args
@given(
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
def test_default(x, default_val):
    with_callable = False
    if x is not None:
        if hasattr(x, "__call__"):
            with_callable = True
        else:
            x_dtype, x = x
            x = x[0].tolist() if isinstance(x, list) else x
    else:
        if hasattr(default_val, "__call__"):
            with_callable = True
        else:
            dv_dtype, default_val = default_val
            default_val = (
                default_val[0].tolist()
                if isinstance(default_val, list)
                else default_val
            )

    truth_val = ivy.to_native(x if x is not None else default_val)
    if with_callable:
        assert ivy.default(x, default_val) == truth_val
    else:
        assert_all_close(
            np.asarray(ivy.default(x, default_val)),
            np.asarray(truth_val),
            rtol=1e-3,
            atol=1e-3,
        )


@handle_cmd_line_args
def test_cache_fn(device):
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


@handle_cmd_line_args
def test_cache_fn_with_args(device):
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


@handle_cmd_line_args
def test_framework_setting_with_threading(device):
    if ivy.current_backend_str() == "jax":
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
        ivy.unset_backend()
        return True

    # get original framework string and array
    fws = ivy.current_backend_str()
    x = ivy.array([0.0, 1.0, 2.0])

    # start jax loop thread
    thread = threading.Thread(target=thread_fn)
    thread.start()
    time.sleep(0.01)
    # start local original framework loop
    ivy.set_backend(fws)
    for _ in range(2000):
        ivy.mean(x)
    ivy.unset_backend()
    assert not thread.join()


@handle_cmd_line_args
def test_framework_setting_with_multiprocessing(device):
    if ivy.current_backend_str() == "numpy":
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
        ivy.unset_backend()
        out_queue.put(True)

    # get original framework string and array
    fws = ivy.current_backend_str()
    x = ivy.array([0.0, 1.0, 2.0])

    # start numpy loop thread
    output_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(target=worker_fn, args=(output_queue,))
    worker.start()

    # start local original framework loop
    ivy.set_backend(fws)
    for _ in range(1000):
        ivy.mean(x)
    ivy.unset_backend()

    worker.join()
    assert output_queue.get_nowait()


@handle_cmd_line_args
def test_explicit_ivy_framework_handles(device):
    if ivy.current_backend_str() == "numpy":
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    # store original framework string and unset
    fw_str = ivy.current_backend_str()
    ivy.unset_backend()

    # set with explicit handle caught
    ivy_exp = ivy.get_backend(fw_str)
    assert ivy_exp.current_backend_str() == fw_str

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
    assert ivy_exp.current_backend_str() == fw_str

    # unset global ivy from numpy
    ivy.unset_backend()


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
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="einops_rearrange"),
)
def test_einops_rearrange(
    dtype_x,
    pattern_and_axes_lengths,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    pattern, axes_lengths = pattern_and_axes_lengths
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="einops_rearrange",
        x=x[0],
        pattern=pattern,
        **axes_lengths,
    )


# einops_reduce
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="einops_reduce"),
)
def test_einops_reduce(
    dtype_x,
    pattern_and_axes_lengths,
    floattypes,
    reduction,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    pattern, axes_lengths = pattern_and_axes_lengths
    dtype, x = dtype_x
    if (reduction in ["mean", "prod"]) and (dtype not in floattypes):
        dtype = ["float32"]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="einops_reduce",
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        pattern=pattern,
        reduction=reduction,
        **axes_lengths,
    )


# einops_repeat
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="einops_repeat"),
)
def test_einops_repeat(
    dtype_x,
    pattern_and_axes_lengths,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    pattern, axes_lengths = pattern_and_axes_lengths
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="einops_repeat",
        x=x[0],
        pattern=pattern,
        **axes_lengths,
    )


# container types
@handle_cmd_line_args
def test_container_types(device):
    cont_types = ivy.container_types()
    assert isinstance(cont_types, list)
    for cont_type in cont_types:
        assert hasattr(cont_type, "keys")
        assert hasattr(cont_type, "values")
        assert hasattr(cont_type, "items")


@handle_cmd_line_args
def test_inplace_arrays_supported(device):
    cur_fw = ivy.current_backend_str()
    if cur_fw in ["numpy", "torch"]:
        assert ivy.inplace_arrays_supported()
    elif cur_fw in ["jax", "tensorflow"]:
        assert not ivy.inplace_arrays_supported()
    else:
        raise Exception("Unrecognized framework")


@handle_cmd_line_args
def test_inplace_variables_supported(device):
    cur_fw = ivy.current_backend_str()
    if cur_fw in ["numpy", "torch", "tensorflow"]:
        assert ivy.inplace_variables_supported()
    elif cur_fw in ["jax"]:
        assert not ivy.inplace_variables_supported()
    else:
        raise Exception("Unrecognized framework")


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_inplace_update(x_val_and_dtypes, tensor_fn, device):
    # ToDo: Ask Daniel about tensor_fn, we use it here since
    #  we don't use helpers.test_function
    x, val = x_val_and_dtypes[1]
    x = tensor_fn(x, dtype="float32", device=device)
    val = tensor_fn(val, dtype="float32", device=device)
    if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or (
        tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()
    ):
        x_inplace = ivy.inplace_update(x, val)
        assert id(x_inplace) == id(x)
        assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(val))
        return


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        num_arrays=2,
        shared_dtype=True,
        min_value=-1e05,
        max_value=1e05,
    ),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_inplace_decrement(x_val_and_dtypes, tensor_fn, device):
    x, val = x_val_and_dtypes[1]
    x, val = x.tolist(), val.tolist()
    x = tensor_fn(x, dtype="float32", device=device)
    val = tensor_fn(val, dtype="float32", device=device)
    new_val = x - val
    if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or (
        tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()
    ):
        x_inplace = ivy.inplace_decrement(x, val)
        assert id(x_inplace) == id(x)
        assert np.allclose(ivy.to_numpy(new_val), ivy.to_numpy(x))
        return


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        num_arrays=2,
        shared_dtype=True,
        min_value=-1e05,
        max_value=1e05,
    ),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_inplace_increment(x_val_and_dtypes, tensor_fn, device):
    x, val = x_val_and_dtypes[1]
    x, val = x.tolist(), val.tolist()
    x = tensor_fn(x, dtype="float32", device=device)
    val = tensor_fn(val, dtype="float32", device=device)
    new_val = x + val
    if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or (
        tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()
    ):
        x_inplace = ivy.inplace_increment(x, val)
        assert id(x_inplace) == id(x)
        assert np.allclose(ivy.to_numpy(new_val), ivy.to_numpy(x_inplace))
        return


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    exclusive=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_ivy_array"),
)
def test_is_ivy_array(
    x_val_and_dtypes,
    exclusive,
    as_variable,
    instance_method,
    num_positional_args,
    native_array,
    container,
    fw,
):
    dtype, x = x_val_and_dtypes
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_ivy_array",
        x=x[0],
        exclusive=exclusive,
    )


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    exclusive=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_array"),
)
def test_is_array(
    x_val_and_dtypes,
    exclusive,
    as_variable,
    num_positional_args,
    instance_method,
    native_array,
    container,
    fw,
):
    dtype, x = x_val_and_dtypes
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_array",
        x=x[0],
        exclusive=exclusive,
    )


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    num_positional_args=helpers.num_positional_args(fn_name="is_ivy_container"),
)
def test_is_ivy_container(
    x_val_and_dtypes,
    as_variable,
    num_positional_args,
    instance_method,
    native_array,
    container,
    fw,
):
    dtype, x = x_val_and_dtypes
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_ivy_container",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=2, max_value=10),
        min_num_dims=1,
    ),
    equality_matrix=st.booleans(),
)
def test_all_equal(
    dtypes_and_xs,
    equality_matrix,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    dtypes, arrays = dtypes_and_xs
    kw = {}
    i = 0
    for x_ in arrays:
        kw["x{}".format(i)] = x_
        i += 1
    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=len(arrays) + 1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="all_equal",
        **kw,
        equality_matrix=equality_matrix,
    )


@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_value=-10,
        max_value=10,
    ),
    max_norm=st.floats(min_value=0.137, max_value=1e05),
    p=st.sampled_from([1, 2, float("inf"), "fro", "nuc"]),
    num_positional_args=helpers.num_positional_args(fn_name="clip_matrix_norm"),
)
def test_clip_matrix_norm(
    dtype_x,
    max_norm,
    p,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="clip_matrix_norm",
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        max_norm=max_norm,
        p=p,
    )


@handle_cmd_line_args
@given(
    val_dtype=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=1,
        allow_nan=True,
        allow_inf=True,
    ),
    include_infs=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="value_is_nan"),
)
def test_value_is_nan(
    val_dtype,
    include_infs,
    as_variable,
    num_positional_args,
    instance_method,
    native_array,
    container,
    fw,
):
    dtype, val = val_dtype
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="value_is_nan",
        x=val,
        include_infs=include_infs,
    )


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_nan=True,
        allow_inf=True,
    ),
    include_infs=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="has_nans"),
)
def test_has_nans(
    x_val_and_dtypes,
    include_infs,
    as_variable,
    num_positional_args,
    instance_method,
    native_array,
    container,
    fw,
):
    dtype, x = x_val_and_dtypes
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="has_nans",
        x=x[0],
        include_infs=include_infs,
    )


@handle_cmd_line_args
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


@handle_cmd_line_args
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


def _composition_1():
    return ivy.relu().argmax()


def _composition_2():
    return ivy.ceil() or ivy.linspace()


# function_supported_devices_and_dtypes
@pytest.mark.parametrize(
    "func, expected",
    [
        (
            _composition_1,
            {
                "cpu": (
                    "bool",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "bfloat16",
                    "float16",
                    "float32",
                    "float64",
                )
            },
        ),
        (
            _composition_2,
            {
                "cpu": (
                    "bool",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "bfloat16",
                    "float16",
                    "float32",
                    "float64",
                )
            },
        ),
    ],
)
def test_function_supported_device_and_dtype(func, expected):
    res = ivy.function_supported_devices_and_dtypes(func)
    exp = {}
    for dev in expected:
        exp[dev] = tuple((set(ivy.valid_dtypes).intersection(expected[dev])))
        if ivy.current_backend_str() == "torch":
            exp[dev] = tuple((set(exp[dev]).difference({"float16"})))

    all_key = set(res.keys()).union(set(exp.keys()))
    for key in all_key:
        assert key in res
        assert key in exp
        assert set(res[key]) == set(exp[key])


# function_unsupported_devices_and_dtypes
@pytest.mark.parametrize(
    "func, expected",
    [
        (_composition_1, {"gpu": ivy.all_dtypes, "tpu": ivy.all_dtypes}),
        # (_composition_2, {'gpu': ivy.all_dtypes, 'tpu': ivy.all_dtypes})
    ],
)
def test_function_unsupported_devices(func, expected):
    res = ivy.function_unsupported_devices_and_dtypes(func)

    exp = expected.copy()

    if ivy.invalid_dtypes:
        exp["cpu"] = ivy.invalid_dtypes
    if ivy.current_backend_str() == "torch":
        exp["cpu"] = tuple((set(exp["cpu"]).union({"float16"})))

    all_key = set(res.keys()).union(set(exp.keys()))
    for key in all_key:
        assert key in res
        assert key in exp
        assert set(res[key]) == set(exp[key])


# Still to Add #
# ---------------#


@handle_cmd_line_args
def test_current_backend_str(fw):
    assert ivy.current_backend_str() == fw


@handle_cmd_line_args
def test_get_min_denominator():
    assert ivy.get_min_denominator() == 1e-12


@handle_cmd_line_args
@given(x=st.floats(allow_nan=False, allow_infinity=False))
def test_set_min_denominator(x):
    ivy.set_min_denominator(x)
    assert ivy.get_min_denominator() == x


@handle_cmd_line_args
def test_get_min_base():
    assert ivy.get_min_base() == 1e-5


@handle_cmd_line_args
@given(x=st.floats(allow_nan=False, allow_infinity=False))
def test_set_min_base(x):
    ivy.set_min_base(x)
    assert ivy.get_min_base() == x


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=3, shared_dtype=True
    ),
    num_positional_args=helpers.num_positional_args(fn_name="stable_divide"),
)
def test_stable_divide(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    instance_method,
    container,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="stable_divide",
        numerator=x[0],
        denominator=x[1],
        min_denominator=x[2],
    )


@st.composite  # ToDo remove when helpers.get_dtypes supports it
def _get_valid_numeric_no_unsigned(draw):
    return list(
        set(draw(helpers.get_dtypes("numeric"))).difference(
            draw(helpers.get_dtypes("unsigned"))
        )
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=pow_helper(available_dtypes=_get_valid_numeric_no_unsigned()),
    dtype_and_min_base=helpers.dtype_and_values(
        available_dtypes=_get_valid_numeric_no_unsigned(),
        num_arrays=1,
        large_abs_safety_factor=100,
        small_abs_safety_factor=100,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="stable_pow"),
)
def test_stable_pow(
    dtypes_and_xs,
    dtype_and_min_base,
    as_variable,
    num_positional_args,
    native_array,
    instance_method,
    container,
    fw,
):
    dtypes, xs = dtypes_and_xs
    input_dtype_min_base, min_base = dtype_and_min_base
    assume(all(["bfloat16" not in x for x in dtypes + input_dtype_min_base]))
    helpers.test_function(
        input_dtypes=dtypes + input_dtype_min_base,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="stable_pow",
        rtol_=1e-2,
        atol_=1e-2,
        base=xs[0][0],
        exponent=np.abs(xs[1]),
        min_base=min_base[0],
    )


@handle_cmd_line_args
def test_get_all_arrays_in_memory():
    return


@handle_cmd_line_args
def test_num_arrays_in_memory():
    return


@handle_cmd_line_args
def test_print_all_arrays_in_memory():
    return


@handle_cmd_line_args
@given(
    x=st.floats(allow_nan=False, allow_infinity=False),
)
def test_set_queue_timeout(x):
    ivy.set_queue_timeout(x)
    ret = ivy.get_queue_timeout()
    assert ret == x


@handle_cmd_line_args
@given(
    x=st.floats(allow_nan=False, allow_infinity=False),
)
def test_get_queue_timeout(x):
    ivy.set_queue_timeout(x)
    ret = ivy.get_queue_timeout()
    assert ret == x


@handle_cmd_line_args
def test_get_tmp_dir():
    ret = ivy.get_tmp_dir()
    assert ret == "/tmp"


@handle_cmd_line_args
def test_set_tmp_dir():
    ivy.set_tmp_dir("/new_dir")
    ret = ivy.get_tmp_dir()
    assert ret == "/new_dir"


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    num_positional_args=helpers.num_positional_args(fn_name="supports_inplace_updates"),
)
def test_supports_inplace_updates(
    x_val_and_dtypes,
    as_variable,
    num_positional_args,
    native_array,
    instance_method,
    container,
    fw,
):
    dtype, x = x_val_and_dtypes
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="supports_inplace_updates",
        test_values=False,
        x=x[0],
    )


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    num_positional_args=helpers.num_positional_args(fn_name="assert_supports_inplace"),
)
def test_assert_supports_inplace(
    x_val_and_dtypes,
    as_variable,
    num_positional_args,
    native_array,
    instance_method,
    container,
    fw,
):
    dtype, x = x_val_and_dtypes
    if fw == "tensorflow" or fw == "jax":
        return
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="assert_supports_inplace",
        ground_truth_backend="numpy",
        x=x[0],
    )


@handle_cmd_line_args
def test_arg_info():
    return


def _fn1(x, y):
    return ivy.matmul(x, y)


def _fn2(x, y):
    return ivy.vecdot(x, y)


def _fn3(x, y):
    ivy.add(x, y)


@given(
    func=st.sampled_from([_fn1, _fn2, _fn3]),
    arrays_and_axes=helpers.arrays_and_axes(
        allow_none=False,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        num=2,
    ),
    in_axes_as_cont=st.booleans(),
)
def test_vmap(func, arrays_and_axes, in_axes_as_cont):

    generated_arrays, in_axes = arrays_and_axes
    arrays = [ivy.native_array(array) for array in generated_arrays]

    if in_axes_as_cont:
        vmapped_func = ivy.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)

    assert callable(vmapped_func)

    try:
        fw_res = vmapped_func(*arrays)
    except Exception:
        fw_res = None

    ivy.set_backend("jax")
    arrays = [ivy.native_array(array) for array in generated_arrays]
    if in_axes_as_cont:
        jax_vmapped_func = ivy.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        jax_vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)

    assert callable(jax_vmapped_func)

    try:
        jax_res = jax_vmapped_func(*arrays)
    except Exception:
        jax_res = None

    ivy.unset_backend()

    if fw_res is not None and jax_res is not None:
        assert np.allclose(
            fw_res, jax_res
        ), f"Results from {ivy.current_backend_str()} and jax are not equal"

    elif fw_res is None and jax_res is None:
        pass
    else:
        assert False, "One of the results is None while other isn't"
