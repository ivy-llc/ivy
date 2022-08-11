"""Collection of tests for unified general functions."""

# global
import time
import einops
import jax.numpy as jnp
import pytest
from hypothesis import given, assume, strategies as st
import numpy as np
from numbers import Number
from collections.abc import Sequence
import torch.multiprocessing as multiprocessing

# local
import threading
import ivy
import ivy.functional.backends.jax
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
import ivy.functional.backends.mxnet
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np

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
@given(fw_str=st.sampled_from(["numpy", "jax", "torch", "mxnet"]))
def test_set_framework(fw_str, device, call):
    ivy.set_backend(fw_str)
    ivy.unset_backend()


# use_framework
def test_use_within_use_framework(device, call):
    with ivy.functional.backends.numpy.use:
        pass
    with ivy.functional.backends.jax.use:
        pass
    with ivy.functional.backends.tensorflow.use:
        pass
    with ivy.functional.backends.torch.use:
        pass
    with ivy.functional.backends.mxnet.use:
        pass


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


def test_get_referrers_recursive(device, call):
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


# copy array
@given(dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes))
def test_copy_array(dtype_and_x, device, call, fw):
    dtype, x = dtype_and_x
    assume(not (fw == "torch" and dtype in ["uint16", "uint32", "uint64"]))

    # mxnet does not support int16
    assume(not (fw == "mxnet" and dtype == "int16"))

    # smoke test
    x = ivy.array(x, dtype=dtype, device=device)
    ret = ivy.copy_array(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    helpers.assert_all_close(ivy.to_numpy(ret), ivy.to_numpy(x))
    assert id(x) != id(ret)


# array_equal
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes, num_arrays=2
    )
)
def test_array_equal(x0_n_x1_n_res, device, call, fw):
    dtype0, x0 = x0_n_x1_n_res[0][0], x0_n_x1_n_res[1][0]
    dtype1, x1 = x0_n_x1_n_res[0][1], x0_n_x1_n_res[1][1]

    # torch does not support those dtypes
    assume(
        not (
            fw == "torch"
            and (
                dtype0 in ["uint16", "uint32", "uint64"]
                or dtype1 in ["uint16", "uint32", "uint64"]
            )
        )
    )

    # mxnet does not support int16, and does not support
    # bool for broadcast_equal method used
    assume(
        not (
            fw == "mxnet"
            and (dtype0 in ["int16", "bool"] or dtype1 in ["int16", "bool"])
        )
    )

    # smoke test
    x0 = ivy.array(x0, dtype=dtype0, device=device)
    x1 = ivy.array(x1, dtype=dtype1, device=device)
    res = ivy.array_equal(x0, x1)
    # type test
    assert ivy.is_ivy_array(x0)
    assert ivy.is_ivy_array(x1)
    assert isinstance(res, bool) or ivy.is_ivy_array(res)
    # value test
    assert res == np.array_equal(np.array(x0, dtype=dtype0), np.array(x1, dtype=dtype1))


# arrays_equal
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes, num_arrays=3
    )
)
def test_arrays_equal(x0_n_x1_n_res, device, call, fw):
    dtype0, x0 = x0_n_x1_n_res[0][0], x0_n_x1_n_res[1][0]
    dtype1, x1 = x0_n_x1_n_res[0][1], x0_n_x1_n_res[1][1]
    dtype2, x2 = x0_n_x1_n_res[0][2], x0_n_x1_n_res[1][2]
    assume(
        not (
            fw == "torch"
            and (
                dtype0 in ["uint16", "uint32", "uint64"]
                or dtype1 in ["uint16", "uint32", "uint64"]
                or dtype2 in ["uint16", "uint32", "uint64"]
            )
        )
    )
    # torch does not support those dtypes
    assume(
        not (
            fw == "mxnet"
            and (dtype0 in ["int16", "bool"] or dtype1 in ["int16", "bool"])
        )
    )
    # mxnet does not support int16, and does not support bool
    # for broadcast_equal method used
    # smoke test
    x0 = ivy.array(x0, dtype=dtype0, device=device)
    x1 = ivy.array(x1, dtype=dtype1, device=device)
    x2 = ivy.array(x2, dtype=dtype2, device=device)
    res = ivy.arrays_equal([x0, x1, x2])
    # type test
    assert ivy.is_ivy_array(x0)
    assert ivy.is_ivy_array(x1)
    assert ivy.is_ivy_array(x2)
    assert isinstance(res, bool) or ivy.is_ivy_array(res)
    # value test
    true_res = (
        np.array_equal(ivy.to_numpy(x0), ivy.to_numpy(x1))
        and np.array_equal(ivy.to_numpy(x0), ivy.to_numpy(x2))
        and np.array_equal(ivy.to_numpy(x1), ivy.to_numpy(x2))
    )
    assert res == true_res


# to_numpy
@given(x0_n_x1_n_res=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes))
def test_to_numpy(x0_n_x1_n_res, device, call, fw):
    dtype, object_in = x0_n_x1_n_res
    assume(not (fw == "torch" and (dtype in ["uint16", "uint32", "uint64"])))
    # torch does not support those dtypes
    assume(not (fw == "mxnet" and dtype == "int16"))
    # mxnet does not support int16
    # smoke test
    ret = ivy.to_numpy(ivy.array(object_in, dtype=dtype, device=device))
    # type test
    assert isinstance(ret, np.ndarray)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    helpers.assert_all_close(ret, np.array(object_in).astype(dtype))


# to_scalar
@given(
    object_in=st.sampled_from([[0.0], [[[1]]], [True], [[1.0]]]),
    dtype=st.sampled_from(ivy_np.valid_dtypes),
)
def test_to_scalar(object_in, dtype, device, call, fw):
    assume(not (fw == "torch" and (dtype in ["uint16", "uint32", "uint64"])))
    # torch does not support those dtypes
    assume(not (fw == "mxnet" and dtype == "int16"))
    # mxnet does not support int16
    # smoke test
    ret = ivy.to_scalar(ivy.array(object_in, dtype=dtype, device=device))
    true_val = ivy.to_numpy(ivy.array(object_in, dtype=dtype)).item()
    # type test
    assert isinstance(ret, type(true_val))
    # value test
    assert ivy.to_scalar(ivy.array(object_in, dtype=dtype, device=device)) == true_val


# to_list
@given(x0_n_x1_n_res=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes))
def test_to_list(x0_n_x1_n_res, device, call, fw):
    dtype, object_in = x0_n_x1_n_res
    assume(dtype in ivy.valid_dtypes)
    # smoke test
    arr = ivy.array(object_in, dtype=dtype, device=device)
    ret = ivy.to_list(arr)
    # type test (result won't be a list if input is 0 dimensional
    if arr.ndim != 0:
        assert isinstance(ret, list)
    # cardinality test
    assert _get_shape_of_list(ret) == _get_shape_of_list(object_in)
    # value test
    assert np.allclose(
        np.nan_to_num(
            np.asarray(ivy.to_list(ivy.array(object_in, dtype=dtype, device=device))),
            posinf=np.inf,
            neginf=-np.inf,
        ),
        np.nan_to_num(np.array(object_in).astype(dtype), posinf=np.inf, neginf=-np.inf),
    )


# shape
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    as_tensor=st.booleans(),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_shape(x0_n_x1_n_res, as_tensor, tensor_fn, device, call, fw):
    dtype, object_in = x0_n_x1_n_res
    assume(
        not (
            fw == "torch"
            and (
                dtype in ["uint16", "uint32", "uint64"]
                or (
                    dtype not in ivy_np.valid_float_dtypes
                    and tensor_fn == helpers.var_fn
                )
            )
        )
    )
    # torch does not support those dtypes
    ret = ivy.shape(tensor_fn(object_in, dtype=dtype, device=device), as_tensor)
    # type test
    if as_tensor:
        assert ivy.is_ivy_array(ret)
    else:
        assert isinstance(ret, tuple)

        ret = ivy.array(ret)
    # cardinality test
    assert ret.shape[0] == len(np.asarray(object_in).shape)
    # value test
    assert np.array_equal(
        ivy.to_numpy(ret), np.asarray(np.asarray(object_in).shape, np.int32)
    )


# get_num_dims
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    as_tensor=st.booleans(),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_get_num_dims(x0_n_x1_n_res, as_tensor, tensor_fn, device, call, fw):
    dtype, object_in = x0_n_x1_n_res
    assume(
        not (
            fw == "torch"
            and (
                dtype in ["uint16", "uint32", "uint64"]
                or (
                    dtype not in ivy_np.valid_float_dtypes
                    and tensor_fn == helpers.var_fn
                )
            )
        )
    )
    # torch does not support those dtypes
    ret = ivy.get_num_dims(tensor_fn(object_in, dtype=dtype, device=device), as_tensor)
    # type test
    if as_tensor:
        assert ivy.is_ivy_array(ret)
    else:
        assert isinstance(ret, int)
        ret = ivy.array(ret)
    # cardinality test
    assert list(ret.shape) == []
    # value test
    assert np.array_equal(
        ivy.to_numpy(ret), np.asarray(len(np.asarray(object_in).shape), np.int32)
    )


# clip_vector_norm
@pytest.mark.parametrize(
    "x_max_norm_n_p_val_clipped",
    [
        (-0.5, 0.4, 2.0, -0.4),
        ([1.7], 1.5, 3.0, [1.5]),
        (
            [[0.8, 2.2], [1.5, 0.2]],
            4.0,
            1.0,
            [[0.6808511, 1.8723406], [1.2765958, 0.17021278]],
        ),
        (
            [[0.8, 2.2], [1.5, 0.2]],
            2.5,
            2.0,
            [[0.71749604, 1.9731141], [1.345305, 0.17937401]],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_clip_vector_norm(
    x_max_norm_n_p_val_clipped, dtype, with_out, tensor_fn, device, call
):
    # smoke test
    if call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_max_norm_n_p_val_clipped[0], dtype=dtype, device=device)
    max_norm = x_max_norm_n_p_val_clipped[1]
    p_val = x_max_norm_n_p_val_clipped[2]
    clipped = x_max_norm_n_p_val_clipped[3]
    if with_out:
        out = ivy.zeros(x.shape if len(x.shape) else (1,))
        ret = ivy.clip_vector_norm(x, max_norm, p_val, out=out)
    else:
        ret = ivy.clip_vector_norm(x, max_norm, p_val)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == (x.shape if len(x.shape) else (1,))
    # value test
    assert np.allclose(
        call(ivy.clip_vector_norm, x, max_norm, p_val), np.array(clipped)
    )
    if with_out:
        if not ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            assert ret is out
            assert ret.data is out.data
    # compilation test
    if call is helpers.torch_call:
        # pytorch jit cannot compile global variables, in this case MIN_DENOMINATOR
        return


# floormod
# @given(
#     xy=helpers.dtype_and_values(
#       available_dtypes = ivy_np.valid_numeric_dtypes,
#       n_arrays=2
#     ),
#     as_variable=st.booleans(),
#     with_out=st.booleans(),
#     num_positional_args=helpers.ints(min_value=1, max_value=2),
#     native_array=st.booleans(),
#     container=st.booleans(),
#     instance_method=st.booleans(),
# )
# def test_floormod(
#     xy,
#     as_variable,
#     with_out,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     device,
#     call,
#     fw,
# ):
#     # smoke test
#     dtype = xy[0]
#     x = xy[1][0]
#     divisor = np.abs(xy[1][1])
#     if 0 in divisor:
#         return
#     if fw == "torch" and any(d in ["uint16", "uint32", "uint64"] for d in dtype):
#         return
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         with_out,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "floormod",
#         x=np.asarray(x, dtype=dtype[0]),
#         y=np.asarray(divisor, dtype=dtype[1]),
#     )


# unstack
@pytest.mark.parametrize(
    "x_n_axis", [(1, -1), ([[0.0, 1.0, 2.0]], 0), ([[0.0, 1.0, 2.0]], 1)]
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_unstack(x_n_axis, dtype, tensor_fn, device, call):
    # smoke test
    x, axis = x_n_axis
    if (
        isinstance(x, Number)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype=dtype, device=device)
    ret = ivy.unstack(x, axis)
    # type test
    assert isinstance(ret, list)
    # cardinality test
    axis_val = (
        axis % len(x.shape)
        if (axis is not None and len(x.shape) != 0)
        else len(x.shape) - 1
    )
    if x.shape == ():
        expected_shape = ()
    else:
        expected_shape = list(x.shape)
        expected_shape.pop(axis_val)
    assert ret[0].shape == tuple(expected_shape)
    # value test
    assert np.allclose(
        call(ivy.unstack, x, axis),
        np.asarray(ivy.functional.backends.numpy.unstack(ivy.to_numpy(x), axis)),
    )


# fourier_encode
@pytest.mark.parametrize(
    "x_n_mf_n_nb_n_gt",
    [
        (
            [2.0],
            4.0,
            4,
            [
                [
                    2.0000000e00,
                    1.7484555e-07,
                    9.9805772e-01,
                    -5.2196848e-01,
                    3.4969111e-07,
                    1.0000000e00,
                    -6.2295943e-02,
                    -8.5296476e-01,
                    1.0000000e00,
                ]
            ],
        ),
        (
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [2.0, 4.0],
            4,
            [
                [
                    [
                        1.0000000e00,
                        -8.7422777e-08,
                        -8.7422777e-08,
                        -8.7422777e-08,
                        -8.7422777e-08,
                        -1.0000000e00,
                        -1.0000000e00,
                        -1.0000000e00,
                        -1.0000000e00,
                    ],
                    [
                        2.0000000e00,
                        1.7484555e-07,
                        9.9805772e-01,
                        -5.2196848e-01,
                        -6.0398321e-07,
                        1.0000000e00,
                        -6.2295943e-02,
                        -8.5296476e-01,
                        1.0000000e00,
                    ],
                ],
                [
                    [
                        3.0000000e00,
                        -2.3849761e-08,
                        -2.3849761e-08,
                        -2.3849761e-08,
                        -2.3849761e-08,
                        -1.0000000e00,
                        -1.0000000e00,
                        -1.0000000e00,
                        -1.0000000e00,
                    ],
                    [
                        4.0000000e00,
                        3.4969111e-07,
                        -1.2434989e-01,
                        8.9044148e-01,
                        -1.2079664e-06,
                        1.0000000e00,
                        -9.9223840e-01,
                        4.5509776e-01,
                        1.0000000e00,
                    ],
                ],
                [
                    [
                        5.0000000e00,
                        -6.7553248e-07,
                        -6.7553248e-07,
                        -6.7553248e-07,
                        -6.7553248e-07,
                        -1.0000000e00,
                        -1.0000000e00,
                        -1.0000000e00,
                        -1.0000000e00,
                    ],
                    [
                        6.0000000e00,
                        4.7699523e-08,
                        -9.8256493e-01,
                        -9.9706185e-01,
                        -3.7192983e-06,
                        1.0000000e00,
                        1.8591987e-01,
                        7.6601014e-02,
                        1.0000000e00,
                    ],
                ],
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_fourier_encode(x_n_mf_n_nb_n_gt, dtype, tensor_fn, device, call):
    # smoke test
    x, max_freq, num_bands, ground_truth = x_n_mf_n_nb_n_gt
    if (
        isinstance(x, Number)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype=dtype, device=device)
    if isinstance(max_freq, list):
        max_freq = tensor_fn(max_freq, dtype=dtype, device=device)
    ret = ivy.fourier_encode(x, max_freq, num_bands)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    x_shape = [1] if x.shape == () else list(x.shape)
    expected_shape = x_shape + [1 + 2 * num_bands]
    assert list(ret.shape) == expected_shape
    # value test
    assert np.allclose(
        call(ivy.fourier_encode, x, max_freq, num_bands),
        np.array(ground_truth),
        atol=1e-5,
    )


# indices_where
@pytest.mark.parametrize("x", [[True], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_indices_where(x, dtype, tensor_fn, device, call):
    # smoke test
    if (
        isinstance(x, Number)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype=dtype, device=device)
    ret = ivy.indices_where(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert len(ret.shape) == 2
    assert ret.shape[-1] == len(x.shape)
    # value test
    assert np.allclose(
        call(ivy.indices_where, x),
        np.asarray(ivy.functional.backends.numpy.indices_where(ivy.to_numpy(x))),
    )


# one_hot
@pytest.mark.parametrize(
    "ind_n_depth", [([0], 1), ([0, 1, 2], 3), ([[1, 3], [0, 0], [8, 4], [7, 9]], 10)]
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_one_hot(ind_n_depth, dtype, tensor_fn, device, call):
    # smoke test
    ind, depth = ind_n_depth
    if (
        isinstance(ind, Number)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    ind = ivy.array(ind, dtype="int32", device=device)
    ret = ivy.one_hot(ind, depth, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == ind.shape + (depth,)
    # value test
    assert np.allclose(
        call(ivy.one_hot, ind, depth, device=device),
        np.asarray(
            ivy.functional.backends.numpy.one_hot(
                ivy.to_numpy(ind), depth, device=device
            )
        ),
    )


# cumsum
@pytest.mark.parametrize(
    "x_n_axis",
    [
        ([[0.0, 1.0, 2.0]], -1),
        ([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], 0),
        ([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], 1),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_cumsum(x_n_axis, dtype, with_out, tensor_fn, device, call):
    # smoke test
    x, axis = x_n_axis
    x = ivy.array(x, dtype=dtype, device=device)
    if with_out:
        if ivy.exists(axis):
            out = ivy.zeros(x.shape)
            ret = ivy.cumsum(x, axis, out=out)
        else:
            out = ivy.zeros(ivy.reshape(x, (-1,)).shape)
            ret = ivy.cumsum(x, axis, out=out)
    else:
        ret = ivy.cumsum(x, axis)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(
        call(ivy.cumsum, x, axis),
        np.asarray(ivy.functional.backends.numpy.cumsum(ivy.to_numpy(x), axis)),
    )
    # out test
    if with_out:
        if not ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            assert ret is out
            assert ret.data is out.data


# cumprod
@pytest.mark.parametrize(
    "x_n_axis",
    [
        ([[0.0, 1.0, 2.0]], -1),
        ([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], 0),
        ([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], 1),
    ],
)
@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_cumprod(x_n_axis, exclusive, dtype, with_out, tensor_fn, device, call):
    # smoke test
    x, axis = x_n_axis
    x = ivy.array(x, dtype=dtype, device=device)
    if with_out:
        if ivy.exists(axis):
            out = ivy.zeros(x.shape)
            ret = ivy.cumprod(x, axis, exclusive=exclusive, out=out)
        else:
            out = ivy.zeros(ivy.reshape(x, (-1,)).shape)
            ret = ivy.cumprod(x, axis, exclusive=exclusive, out=out)
    else:
        ret = ivy.cumprod(x, axis, exclusive)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(
        call(ivy.cumprod, x, axis, exclusive),
        np.asarray(
            ivy.functional.backends.numpy.cumprod(ivy.to_numpy(x), axis, exclusive)
        ),
    )
    # out test
    if with_out:
        if not ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            assert ret is out
            assert ret.data is out.data


# scatter_flat
@pytest.mark.parametrize(
    "inds_n_upd_n_size_n_tnsr_n_wdup",
    [
        ([0, 4, 1, 2], [1, 2, 3, 4], 8, None, False),
        ([0, 4, 1, 2, 0], [1, 2, 3, 4, 5], 8, None, True),
        ([0, 4, 1, 2, 0], [1, 2, 3, 4, 5], None, [11, 10, 9, 8, 7, 6], True),
    ],
)
@pytest.mark.parametrize("red", ["sum", "min", "max", "replace"])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_scatter_flat(inds_n_upd_n_size_n_tnsr_n_wdup, red, dtype, tensor_fn, call):
    # smoke test
    if red in ("sum", "min", "max") and call is helpers.mx_call:
        # mxnet does not support sum, min or max reduction for scattering
        pytest.skip()
    inds, upd, size, tensor, with_duplicates = inds_n_upd_n_size_n_tnsr_n_wdup
    if ivy.exists(tensor) and call is helpers.mx_call:
        # mxnet does not support scattering into pre-existing tensors
        pytest.skip()
    inds = ivy.array(inds, dtype="int32")
    upd = tensor_fn(upd, dtype=dtype)
    if tensor:
        # pytorch variables do not support in-place updates
        tensor = (
            ivy.array(tensor, dtype=dtype)
            if ivy.current_backend_str() == "torch"
            else tensor_fn(tensor, dtype=dtype)
        )
    ret = ivy.scatter_flat(inds, upd, size, tensor, red)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    if size:
        assert ret.shape == (size,)
    else:
        assert ret.shape == tensor.shape
    # value test
    if red == "replace" and with_duplicates:
        # replace with duplicates give non-deterministic outputs
        return
    inds_np = ivy.to_numpy(inds)
    upd_np = ivy.to_numpy(upd)
    tensor_np = ivy.to_numpy(tensor) if ivy.exists(tensor) else tensor
    with ivy.functional.backends.numpy.use:
        true_np = ivy.scatter_flat(inds_np, upd_np, size, tensor_np, red).data
    assert np.allclose(
        call(ivy.scatter_flat, inds, upd, size, tensor, red),
        true_np,
    )


# scatter_nd
@pytest.mark.parametrize(
    "inds_n_upd_n_shape_tnsr_n_wdup",
    [
        ([[4], [3], [1], [7]], [9, 10, 11, 12], [8], None, False),
        ([[0, 1, 2]], [1], [3, 3, 3], None, False),
        (
            [[0], [2]],
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            ],
            [4, 4, 4],
            None,
            False,
        ),
        (
            [[0, 1, 2]],
            [1],
            None,
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[4, 5, 6], [7, 8, 9], [1, 2, 3]],
                [[7, 8, 9], [1, 2, 3], [4, 5, 6]],
            ],
            False,
        ),
    ],
)
@pytest.mark.parametrize("red", ["sum", "min", "max", "replace"])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_scatter_nd(inds_n_upd_n_shape_tnsr_n_wdup, red, dtype, tensor_fn, call):
    # smoke test
    if red in ("sum", "min", "max") and call is helpers.mx_call:
        # mxnet does not support sum, min or max reduction for scattering
        pytest.skip()
    inds, upd, shape, tensor, with_duplicates = inds_n_upd_n_shape_tnsr_n_wdup
    if ivy.exists(tensor) and call is helpers.mx_call:
        # mxnet does not support scattering into pre-existing tensors
        pytest.skip()
    inds = ivy.array(inds, dtype="int32")
    upd = tensor_fn(upd, dtype=dtype)
    if tensor:
        # pytorch variables do not support in-place updates
        tensor = (
            ivy.array(tensor, dtype=dtype)
            if ivy.current_backend_str() == "torch"
            else tensor_fn(tensor, dtype=dtype)
        )
    inds_np = ivy.to_numpy(inds)
    upd_np = ivy.to_numpy(upd)
    tensor_np = ivy.to_numpy(tensor) if ivy.exists(tensor) else tensor
    ret = ivy.scatter_nd(inds, upd, shape, tensor, red)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    if shape:
        assert tuple(ret.shape) == tuple(shape)
    else:
        assert tuple(ret.shape) == tuple(tensor.shape)
    # value test
    if red == "replace" and with_duplicates:
        # replace with duplicates give non-deterministic outputs
        return
    with ivy.functional.backends.numpy.use:
        true_np = ivy.scatter_nd(inds_np, upd_np, shape, tensor_np, red).data
    assert np.allclose(ret, true_np)


# gather
@pytest.mark.parametrize(
    "prms_n_inds_n_axis",
    [
        ([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [0, 4, 7], 0),
        ([[1, 2], [3, 4]], [[0, 0], [1, 0]], 1),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_gather(prms_n_inds_n_axis, dtype, with_out, tensor_fn, call):
    # smoke test
    prms, inds, axis = prms_n_inds_n_axis
    prms = tensor_fn(prms, dtype=dtype)
    inds = ivy.array(inds, dtype="int32")
    if with_out:
        out = ivy.zeros(inds.shape)
        ret = ivy.gather(prms, inds, axis, out=out)
    else:
        ret = ivy.gather(prms, inds, axis)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == inds.shape
    # value test
    assert np.allclose(
        call(ivy.gather, prms, inds, axis),
        np.asarray(
            ivy.functional.backends.numpy.gather(
                ivy.to_numpy(prms), ivy.to_numpy(inds), axis
            )
        ),
    )
    # out test
    if with_out:
        if not ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            assert ret is out
            assert ret.data is out.data


# gather_nd
@pytest.mark.parametrize(
    "prms_n_inds",
    [
        ([[[0.0, 1.0], [2.0, 3.0]], [[0.1, 1.1], [2.1, 3.1]]], [[0, 1], [1, 0]]),
        ([[[0.0, 1.0], [2.0, 3.0]], [[0.1, 1.1], [2.1, 3.1]]], [[[0, 1]], [[1, 0]]]),
        (
            [[[0.0, 1.0], [2.0, 3.0]], [[0.1, 1.1], [2.1, 3.1]]],
            [[[0, 1, 0]], [[1, 0, 1]]],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_gather_nd(prms_n_inds, dtype, tensor_fn, call):
    # smoke test
    prms, inds = prms_n_inds
    prms = tensor_fn(prms, dtype=dtype)
    inds = ivy.array(inds, dtype="int32")
    ret = ivy.gather_nd(prms, inds)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == inds.shape[:-1] + prms.shape[inds.shape[-1] :]
    # value test
    assert np.allclose(
        call(ivy.gather_nd, prms, inds),
        np.asarray(
            ivy.functional.backends.numpy.gather_nd(
                ivy.to_numpy(prms), ivy.to_numpy(inds)
            )
        ),
    )


# exists
@pytest.mark.parametrize("x", [[1.0], None, [[10.0, 9.0, 8.0]]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_exists(x, dtype, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype=dtype, device=device) if x is not None else None
    ret = ivy.exists(x)
    # type test
    assert isinstance(ret, bool)
    # value test
    y_true = x is not None
    assert ret == y_true


# default
@pytest.mark.parametrize(
    "x_n_dv", [([1.0], [2.0]), (None, [2.0]), ([[10.0, 9.0, 8.0]], [2.0])]
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_default(x_n_dv, dtype, tensor_fn, device, call):
    x, dv = x_n_dv
    # smoke test
    x = tensor_fn(x, dtype=dtype, device=device) if x is not None else None
    dv = tensor_fn(dv, dtype=dtype, device=device)
    ret = ivy.default(x, dv)
    # type test
    assert ivy.is_ivy_array(ret)
    # value test
    y_true = ivy.to_numpy(x if x is not None else dv)
    assert np.allclose(call(ivy.default, x, dv), y_true)


def test_cache_fn(device, call):
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


def test_cache_fn_with_args(device, call):
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


def test_framework_setting_with_threading(device, call):
    if call is helpers.jnp_call:
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


def test_framework_setting_with_multiprocessing(device, call):

    if call is helpers.np_call:
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


def test_explicit_ivy_framework_handles(device, call):

    if call is helpers.np_call:
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
@pytest.mark.parametrize(
    "x_n_pattern_n_newx",
    [([[0.0, 1.0, 2.0, 3.0]], "b n -> n b", [[0.0], [1.0], [2.0], [3.0]])],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_einops_rearrange(x_n_pattern_n_newx, dtype, tensor_fn, device, call):
    # smoke test
    x, pattern, new_x = x_n_pattern_n_newx
    x = tensor_fn(x, dtype=dtype, device=device)
    ret = ivy.einops_rearrange(x, pattern)
    true_ret = einops.rearrange(ivy.to_native(x), pattern)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert list(ret.shape) == list(true_ret.shape)
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(true_ret))


# einops_reduce
@pytest.mark.parametrize(
    "x_n_pattern_n_red_n_newx", [([[0.0, 1.0, 2.0, 3.0]], "b n -> b", "mean", [1.5])]
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_einops_reduce(x_n_pattern_n_red_n_newx, dtype, tensor_fn, device, call):
    # smoke test
    x, pattern, reduction, new_x = x_n_pattern_n_red_n_newx
    x = tensor_fn(x, dtype=dtype, device=device)
    ret = ivy.einops_reduce(x, pattern, reduction)
    true_ret = einops.reduce(ivy.to_native(x), pattern, reduction)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert list(ret.shape) == list(true_ret.shape)
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(true_ret))


# einops_repeat
@pytest.mark.parametrize(
    "x_n_pattern_n_al_n_newx",
    [
        (
            [[0.0, 1.0, 2.0, 3.0]],
            "b n -> b n c",
            {"c": 2},
            [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]],
        )
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_einops_repeat(x_n_pattern_n_al_n_newx, dtype, tensor_fn, device, call):
    # smoke test
    x, pattern, axes_lengths, new_x = x_n_pattern_n_al_n_newx
    x = tensor_fn(x, dtype=dtype, device=device)
    ret = ivy.einops_repeat(x, pattern, **axes_lengths)
    true_ret = einops.repeat(ivy.to_native(x), pattern, **axes_lengths)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert list(ret.shape) == list(true_ret.shape)
    # value test
    assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(true_ret))


# container types
def test_container_types(device, call):
    cont_types = ivy.container_types()
    assert isinstance(cont_types, list)
    for cont_type in cont_types:
        assert hasattr(cont_type, "keys")
        assert hasattr(cont_type, "values")
        assert hasattr(cont_type, "items")


def test_inplace_arrays_supported(device, call):
    cur_fw = ivy.current_backend_str()
    if cur_fw in ["numpy", "mxnet", "torch"]:
        assert ivy.inplace_arrays_supported()
    elif cur_fw in ["jax", "tensorflow"]:
        assert not ivy.inplace_arrays_supported()
    else:
        raise Exception("Unrecognized framework")


def test_inplace_variables_supported(device, call):
    cur_fw = ivy.current_backend_str()
    if cur_fw in ["numpy", "mxnet", "torch", "tensorflow"]:
        assert ivy.inplace_variables_supported()
    elif cur_fw in ["jax"]:
        assert not ivy.inplace_variables_supported()
    else:
        raise Exception("Unrecognized framework")


@pytest.mark.parametrize("x_n_new", [([0.0, 1.0, 2.0], [2.0, 1.0, 0.0]), (0.0, 1.0)])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_inplace_update(x_n_new, tensor_fn, device, call):
    x_orig, new_val = x_n_new
    if call is helpers.mx_call and isinstance(x_orig, Number):
        # MxNet supports neither 0-dim variables nor 0-dim inplace updates
        pytest.skip()
    x_orig = tensor_fn(x_orig, dtype="float32", device=device)
    new_val = tensor_fn(new_val, dtype="float32", device=device)
    if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or (
        tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()
    ):
        x = ivy.inplace_update(x_orig, new_val)
        assert id(x) == id(x_orig)
        assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(new_val))
        return
    pytest.skip()


@pytest.mark.parametrize("x_n_dec", [([0.0, 1.0, 2.0], [2.0, 1.0, 0.0]), (0.0, 1.0)])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_inplace_decrement(x_n_dec, tensor_fn, device, call):
    x_orig, dec = x_n_dec
    if call is helpers.mx_call and isinstance(x_orig, Number):
        # MxNet supports neither 0-dim variables nor 0-dim inplace updates
        pytest.skip()
    x_orig = tensor_fn(x_orig, dtype="float32", device=device)
    dec = tensor_fn(dec, dtype="float32", device=device)
    new_val = x_orig - dec
    if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or (
        tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()
    ):
        x = ivy.inplace_decrement(x_orig, dec)
        assert id(x) == id(x_orig)
        assert np.allclose(ivy.to_numpy(new_val), ivy.to_numpy(x))
        return
    pytest.skip()


@pytest.mark.parametrize("x_n_inc", [([0.0, 1.0, 2.0], [2.0, 1.0, 0.0]), (0.0, 1.0)])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_inplace_increment(x_n_inc, tensor_fn, device, call):
    x_orig, inc = x_n_inc
    if call is helpers.mx_call and isinstance(x_orig, Number):
        # MxNet supports neither 0-dim variables nor 0-dim inplace updates
        pytest.skip()
    x_orig = tensor_fn(x_orig, dtype="float32", device=device)
    inc = tensor_fn(inc, dtype="float32", device=device)
    new_val = x_orig + inc
    if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or (
        tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()
    ):
        x = ivy.inplace_increment(x_orig, inc)
        assert id(x) == id(x_orig)
        assert np.allclose(ivy.to_numpy(new_val), ivy.to_numpy(x))
        return
    pytest.skip()


# Still to Add #
# ---------------#

# is_ivy_array
# is_array
# is_ivy_container
# all_equal
# to_numpy
# clip_matrix_norm
# unstack
# value_is_nan
# has_nans
# exists
# shape_to_tuple
# try_else_none
# arg_names
# cache_fn
# current_framework_str
# get_min_denominator
# set_min_denominator
# get_min_base
# set_min_base
# stable_divide
# stable_pow
# get_all_arrays_in_memory
# num_arrays_in_memory
# print_all_arrays_in_memory
# set_queue_timeout
# queue_timeout
# tmp_dir
# set_tmp_dir
# supports_inplace
# assert_supports_inplace
# arg_info
