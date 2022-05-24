"""Collection of tests for utility functions."""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# all
@pytest.mark.parametrize("x", [[1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]]])
@pytest.mark.parametrize("axis", [None, 0, -1, (0,), (-1,)])
@pytest.mark.parametrize("kd", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_all(x, axis, kd, dtype, with_out, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    if axis is None:
        expected_shape = [1] * len(x.shape) if kd else []
    else:
        axis_ = [axis] if isinstance(axis, int) else axis
        axis_ = [item % len(x.shape) for item in axis_]
        expected_shape = list(x.shape)
        if kd:
            expected_shape = [
                1 if i % len(x.shape) in axis_ else item
                for i, item in enumerate(expected_shape)
            ]
        else:
            [expected_shape.pop(item) for item in axis_]
    if with_out:
        out = ivy.astype(ivy.zeros(tuple(expected_shape)), ivy.bool)
        ret = ivy.all(x, axis, kd, out=out)
    else:
        ret = ivy.all(x, axis, kd)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == tuple(expected_shape)
    if with_out:
        if not ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            assert ret is out
            assert ret.data is out.data
            # value test
    assert np.allclose(
        call(ivy.all, x), ivy.functional.backends.numpy.all(ivy.to_numpy(x))
    )


# any
@pytest.mark.parametrize("x", [[1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]]])
@pytest.mark.parametrize("axis", [None, 0, -1, (0,), (-1,)])
@pytest.mark.parametrize("kd", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_any(x, axis, kd, dtype, with_out, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    if axis is None:
        expected_shape = [1] * len(x.shape) if kd else []
    else:
        axis_ = [axis] if isinstance(axis, int) else axis
        axis_ = [item % len(x.shape) for item in axis_]
        expected_shape = list(x.shape)
        if kd:
            expected_shape = [
                1 if i % len(x.shape) in axis_ else item
                for i, item in enumerate(expected_shape)
            ]
        else:
            [expected_shape.pop(item) for item in axis_]
    if with_out:
        out = ivy.astype(ivy.zeros(tuple(expected_shape)), ivy.bool)
        ret = ivy.any(x, axis, kd, out=out)
    else:
        ret = ivy.any(x, axis, kd)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == tuple(expected_shape)
    if with_out:
        if not ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            assert ret is out
            assert ret.data is out.data
            # value test
    assert np.allclose(
        call(ivy.any, x), ivy.functional.backends.numpy.any(ivy.to_numpy(x))
    )
