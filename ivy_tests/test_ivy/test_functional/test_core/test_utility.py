"""Collection of tests for utility functions."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


# all
# @pytest.mark.parametrize("x", [[1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]]])
# @pytest.mark.parametrize("axis", [None, 0, -1, (0,), (-1,)])
# @pytest.mark.parametrize("kd", [True, False])
# @pytest.mark.parametrize("dtype", ["float32"])
# @pytest.mark.parametrize("with_out", [True, False])
# @pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
@given(
    data=st.data(),
    kd=st.booleans(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    with_out=st.booleans(),
    as_variable=st.booleans()
)
def test_all(data, kd, dtype, with_out, as_variable, device, call):
    # smoke test
    if dtype == 'float16':
        return
    x, axis = data.draw(helpers.get_axis(dtype))
    x = ivy.array(x, dtype=dtype, device=device)
    if as_variable:
        x = ivy.variable(x)
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
        out = ivy.astype(ivy.zeros(tuple(expected_shape)), dtype=ivy.bool)
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
@given(
    data=st.data(),
    kd=st.booleans(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    with_out=st.booleans(),
    as_variable=st.booleans()
)
def test_any(data, kd, dtype, with_out, as_variable, device, call):
    # smoke test
    # print(ivy_np.invalid_float_dtypes)
    if dtype in ivy_np.invalid_float_dtypes:
        return
    x, axis = data.draw(helpers.get_axis(dtype))
    x = ivy.array(x, dtype=dtype, device=device)
    if as_variable:
        x = ivy.variable(x)
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
        out = ivy.astype(ivy.zeros(tuple(expected_shape)), dtype=ivy.bool)
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
