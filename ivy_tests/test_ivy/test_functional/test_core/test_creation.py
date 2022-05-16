"""Collection of tests for creation functions."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np

# array
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_dtype_strs),
    from_numpy=st.booleans(),
)
def test_array(dtype_and_x, from_numpy, device, call, fw):
    dtype, object_in = dtype_and_x
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return
    if call in [helpers.mx_call] and dtype == "int16":
        # mxnet does not support int16
        return
    # to numpy
    if from_numpy:
        object_in = np.array(object_in)
    # smoke test
    ret = ivy.array(object_in, dtype, device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    helpers.assert_all_close(ivy.to_numpy(ret), np.array(object_in).astype(dtype))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return


# native_array
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_dtype_strs),
    from_numpy=st.booleans(),
)
def test_native_array(dtype_and_x, from_numpy, device, call, fw):
    dtype, object_in = dtype_and_x
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return
    if call in [helpers.mx_call] and dtype == "int16":
        # mxnet does not support int16
        return
    # to numpy
    if from_numpy:
        object_in = np.array(object_in)
    # smoke test
    ret = ivy.native_array(object_in, dtype, device)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    helpers.assert_all_close(ivy.to_numpy(ret), np.array(object_in).astype(dtype))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return
